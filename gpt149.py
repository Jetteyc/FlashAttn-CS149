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
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler
from flash_attn import flash_attn_func
import os
import shutil


DEBUG = False
print("\nCompiling code into a PyTorch module...\n\n")

if os.path.exists('./build'):
    shutil.rmtree('./build')
os.makedirs('./build')

# mr = load(
#     name="custom_module",
#     sources=["module.cpp", "kernel.cu"],
#     extra_cuda_cflags=["-arch=sm_86", "-g", "-G"],
#     build_directory='./build',
#     verbose=True
# )
mr = load(
    name="custom_module",
    sources=["module.cpp", "kernel.cu"],
    extra_cuda_cflags=["-arch=sm_86"],
    build_directory='./build',
    verbose=False
)
class CustomAttention(nn.Module):
    def __init__(self, Q, K, V, Q_FA, K_FA, V_FA, B, H, N, d, isRef=False, bc=256, br=256):
        super(nn.Module, self).__init__()
        self.Q=Q
        self.K=K
        self.V=V
        self.Q_FA = Q_FA
        self.K_FA = K_FA
        self.V_FA = V_FA
        self.B=B
        self.H=H
        self.N=N
        self.d=d
        self.isRef=isRef
        self.bc=bc
        self.br=br

    def myFA1(self):
        if self.Q is not None:
            device = self.Q.device
        else:
            device = self.Q_FA.device
        d = self.d
        L = torch.zeros((self.B, self.H, self.N), device=device, dtype=torch.float16)
        M = torch.zeros((self.B, self.H, self.N), device=device, dtype=torch.float16)
        if self.isRef:
            with record_function("REFERENCE - FLASH ATTENTION"):
                out = flash_attn_func(self.Q_FA, self.K_FA, self.V_FA)
                # out = badSoftmax(self.Q, self.K, self.V)
            return out
        with record_function("STUDENT - FLASH ATTENTION - v1"):
            Q = self.Q.contiguous()
            K = self.K.contiguous()
            V = self.V.contiguous()
            out = mr.myFA1(Q, K, V, L, M, self.bc, self.br, self.B, self.H, self.N, d)
        return out
    
def createQKVSimple(B, H, N, d, device="cuda"):
    Q = torch.empty(B, H, N, d, device="cpu", dtype=torch.float16)
    K = torch.empty(B, H, N, d, device="cpu", dtype=torch.float16)
    V = torch.empty(B, H, N, d, device="cpu", dtype=torch.float16)
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



def trace_handler(p):
    output = p.key_averages().table(sort_by="cuda_time_total", row_limit=10)
    print(output)
    # p.export_chrome_trace("trace_" + str(p.step_num) + ".json")
    tb_handler = tensorboard_trace_handler("tb_logs")
    tb_handler(p)

def testTemplate(customFunc, params, is_fa_ref=False, running_times=5):
    start = time.time()
    B, H, N, d = params
    Q, K, V = createQKVSimple(B, H, N, d)
    res_ref = badSoftmax(Q, K, V)
    end = time.time()
    pytorch_time = end - start
    print(f"pytorch_time: {pytorch_time}")

    if not DEBUG:
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
            record_shapes=True, profile_memory=True,
            with_stack=True, with_modules=True, with_flops=True,
            on_trace_ready=trace_handler
        ) as p:
            for i in range(running_times):
                res = customFunc()
                p.step()
    
    else:
        res = customFunc()
        
        
    res_ref_cpu = res_ref.cpu().clone()
    if is_fa_ref == True: 
        res = res.transpose(1, 2)
    res_cpu = res.cpu().clone()
    
    # print("res_ref",res_ref_cpu)
    # print("res",res_cpu)

    torch.allclose(res_ref_cpu, res_cpu, atol=1e-2, rtol=1e-4)


def mytest_simple():
    A = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float16)
    B = torch.tensor([4.0, 5.0, 6.0], device="cuda", dtype=torch.float16)
    C = mr.mytest(A, B)
    
    expected = torch.tensor([5.0, 7.0, 9.0], device="cuda", dtype=torch.float16)
    assert torch.allclose(C, expected, atol=1e-3), f"Test failed! Expected {expected}, got {C}"
    print("Test passed! Result:", C)

def fa1Test(B, H, N, d, bc, br, running_times=5):
    print("Running Test: Flash Attention - 1\n")
    # shape1
    # N, d, B, H = 512, 32, 1, 4
    Q,K,V = createQKVSimple(B, H, N, d)
    if DEBUG:
        Q_cpu = Q.cpu().clone()
        K_cpu = K.cpu().clone()
        V_cpu = V.cpu().clone()
        print("Q ", Q_cpu)
        print("K ", K_cpu)
        print("V ", V_cpu)
    Q_FA = Q.transpose(1, 2) # B, N, H, d
    K_FA = K.transpose(1, 2)
    V_FA = V.transpose(1, 2)
    params = (B, H, N, d)
    attentionModuleStudent = CustomAttention(Q,K,V, None, None, None, B, H, N, d, False, bc, br)
    attentionModuleReference = CustomAttention(None, None, None, Q_FA, K_FA, V_FA, B, H, N, d, True, bc, br)
    print(f"-----RUNNING REFERENCE IMPLEMENTATION-----\n")
    testTemplate(attentionModuleReference.myFA1, params, True)
    time.sleep(3)
    print(f"-----RUNNING STUDENT IMPLEMENTATION-----\n")
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
            fa1Test(B, H, N, d, int(args.bc), int(args.br))
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)
        
if __name__ == "__main__":
    main()
