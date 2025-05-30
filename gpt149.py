import argparse
import time
import math
import random
import inspect
from dataclasses import dataclass
import sys, getopt
from os import getcwd, path
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_func
import os
import shutil


DEBUG = False
print("\nCompiling code into a PyTorch module...\n\n")

if os.path.exists('./build'):
    shutil.rmtree('./build')
os.makedirs('./build')

mr = load(
    name="custom_module",
    sources=["module.cpp", "kernel.cu"],
    extra_cuda_cflags=["-arch=sm_86", "-g", "-G"],
    build_directory='./build',
    verbose=True
)

class CustomAttention(nn.Module):
    def __init__(self, Q, K, V, B, H, N, d, isRef=False, bc=256, br=256):
        super(nn.Module, self).__init__()
        self.Q=Q
        self.K=K
        self.V=V
        self.B=B
        self.H=H
        self.N=N
        self.d=d
        self.isRef=isRef
        self.bc=bc
        self.br=br

    def myFA1(self):
        device = self.Q.device
        d = self.d
        L = torch.zeros((self.B, self.H, self.N), device=device)
        M = torch.zeros((self.B, self.H, self.N), device=device)
        if self.isRef:
            with record_function("REFERENCE - FLASH ATTENTION"):
                Q = self.Q.transpose(1, 2) # B, N, H, d
                K = self.K.transpose(1, 2)
                V = self.V.transpose(1, 2)
                out = flash_attn_func(Q, K, V).transpose(1, 2)
                # out = badSoftmax(self.Q, self.K, self.V)
            return out
        with record_function("STUDENT - FLASH ATTENTION - v1"):
            Q = self.Q.contiguous()
            K = self.K.contiguous()
            V = self.V.contiguous()
            out = mr.myFA1(Q, K, V, L, M, self.bc, self.br, self.B, self.H, self.N, d)
        return out
    
def createQKVSimple(B, H, N, d, device="cuda"):
    Q = torch.empty(B, H, N, d, device="cpu")
    K = torch.empty(B, H, N, d, device="cpu")
    V = torch.empty(B, H, N, d, device="cpu")
    for b in range(B):
        for h in range(H):
            for i in range(N):
                for j in range(d):
                    Q[b][h][i][j] = 0.0002 * i + 0.0001 * j
                    K[b][h][i][j] = 0.0006 * i + 0.0003 * j
                    V[b][h][i][j] = 0.00015 * i + 0.0008 * j
    return Q.to(device), K.to(device), V.to(device)

def badSoftmax(Q, K, V):
    # 输入形状为 (B, H, N, d)
    d = Q.shape[-1]
    QK = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(d))

    P = torch.exp(QK)
    Lij = P.sum(dim=-1, keepdim=True)
    softmax_QK = F.softmax(QK, dim=-1)
    
    if DEBUG:
        QK_cpu = QK.cpu().clone()
        print("Sij", QK_cpu)
        P_cpu = P.cpu().clone()
        print("Pij", P_cpu)
        Lij_cpu = Lij.cpu().clone()
        print("lij", Lij_cpu)

    QKV = softmax_QK @ V

    return QKV

def testTemplate(customFunc, params):
    start = time.time()
    B, H, N, d = params
    Q, K, V = createQKVSimple(B, H, N, d)
    res_ref = badSoftmax(Q, K, V)
    end = time.time()
    pytorch_time = end - start
    print(f"pytorch_time: {pytorch_time}")
    with torch.autograd.profiler.profile(use_device='cuda') as prof:
        res = customFunc()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))
    res_ref_cpu = res_ref.cpu().clone()
    res_cpu = res.cpu().clone()
    # print("res_ref",res_ref_cpu)
    # print("res",res_cpu)
    diff = torch.abs(res_ref_cpu - res_cpu)
    not_close = diff > 1e-3
    count = not_close.sum().item()
    print(f"Total mismatched elements: {count} / {diff.numel()}\n\n")


def mytest_simple():
    A = torch.tensor([1.0, 2.0, 3.0], device="cuda")
    B = torch.tensor([4.0, 5.0, 6.0], device="cuda")
    C = mr.mytest(A, B)
    
    expected = torch.tensor([5.0, 7.0, 9.0], device="cuda")
    assert torch.allclose(C, expected, atol=1e-3), f"Test failed! Expected {expected}, got {C}"
    print("Test passed! Result:", C)

def fa1Test(B, H, N, d, bc, br, device="cuda"):
    print("Running Test: Flash Attention - 1\n")
    # shape1
    # N, d, B, H = 1024, 32, 1, 4
    Q,K,V = createQKVSimple(B, H, N, d, device=device)
    params = (B, H, N, d)
    attentionModuleStudent = CustomAttention(Q,K,V, B, H, N, d, False, bc, br)
    attentionModuleReference = CustomAttention(Q,K,V, B, H, N, d, True, bc, br)
    # print("-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    # testTemplate(attentionModuleReference.myFA1, params)
    # time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-----\n")
    testTemplate(attentionModuleStudent.myFA1, params)
    time.sleep(3)
    # print("-----RUNNING REFERENCE IMPLEMENTATION-2----\n")
    # testTemplate(attentionModuleReference.myFA1, params)
    # time.sleep(3)
    print("-----RUNNING STUDENT IMPLEMENTATION-2----\n")
    testTemplate(attentionModuleStudent.myFA1, params)


def main():

    d=32
    B=1
    H=4
    
    parser = argparse.ArgumentParser()
    parser.add_argument("testname", default="fa1", help="name of test to run: test, fa1")
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    parser.add_argument("-bc",  default="32", help="Flash Attention Bc Size")
    parser.add_argument("-br", default="32", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Br Size")
    parser.add_argument("--device", default="cuda", help="Device to use: cpu or cuda")

    args = parser.parse_args()

    if args.model == "shakes128":
        N = 128
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes256":
        N = 256
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes1024":
        N = 1024
        model_filename = "out-shakespeare-char2048Good"
    elif args.model == "shakes2048":
        N = 2048
        model_filename = "out-shakespeare-char2048Good"
    else:
        print("Unknown model name: %s" % args.model)
        return
    
    if args.inference == False:
        N = int(args.N)
        if args.testname == "test":
            mytest_simple()
        elif args.testname == "fa1":
            fa1Test(N, d, B, H, int(args.bc), int(args.br), device=args.device)
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

        
if __name__ == "__main__":
    main()
