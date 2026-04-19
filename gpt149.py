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
    extra_cuda_cflags=["-arch=sm_120"],
    build_directory='./build',
    verbose=False
)


class MyFA1Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, bc, br, use_wmma):
        B, H, N, d = Q.shape
        L = torch.zeros((B, H, N), device=Q.device, dtype=Q.dtype)
        M = torch.zeros((B, H, N), device=Q.device, dtype=Q.dtype)
        O = mr.myFA1(
            Q.contiguous(),
            K.contiguous(),
            V.contiguous(),
            L,
            M,
            int(bc),
            int(br),
            B,
            H,
            N,
            d,
            int(use_wmma),
        )
        ctx.save_for_backward(Q, K, V)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V = ctx.saved_tensors
        with torch.enable_grad():
            Q_ref = Q.detach().float().requires_grad_(True)
            K_ref = K.detach().float().requires_grad_(True)
            V_ref = V.detach().float().requires_grad_(True)
            scale = 1.0 / math.sqrt(Q_ref.shape[-1])
            scores = torch.matmul(Q_ref, K_ref.transpose(-2, -1)) * scale
            probs = torch.softmax(scores, dim=-1)
            out_ref = torch.matmul(probs, V_ref)
            dQ, dK, dV = torch.autograd.grad(
                out_ref,
                (Q_ref, K_ref, V_ref),
                grad_outputs=dO.float(),
                retain_graph=False,
                create_graph=False,
            )
        return dQ.to(Q.dtype), dK.to(K.dtype), dV.to(V.dtype), None, None, None


def myFA1_autograd(Q, K, V, bc, br, use_wmma):
    return MyFA1Function.apply(Q, K, V, bc, br, use_wmma)


class CustomAttention(nn.Module):
    def __init__(self, Q, K, V, Q_FA, K_FA, V_FA, B, H, N, d, isRef=False, bc=256, br=256, use_wmma=False):
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
        self.use_wmma=use_wmma

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
            out = myFA1_autograd(Q, K, V, self.bc, self.br, self.use_wmma)
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

def benchmarkCudaOp(customFunc, warmup_iters=5, benchmark_iters=20):
    # Warm up kernels and caches first.
    for _ in range(warmup_iters):
        customFunc()
    torch.cuda.synchronize()

    timings_ms = []
    peak_mem_bytes = 0
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    res = None
    for _ in range(benchmark_iters):
        torch.cuda.reset_peak_memory_stats()
        start_event.record()
        res = customFunc()
        end_event.record()
        torch.cuda.synchronize()
        timings_ms.append(start_event.elapsed_time(end_event))
        peak_mem_bytes = max(peak_mem_bytes, torch.cuda.max_memory_allocated())
    avg_ms = sum(timings_ms) / len(timings_ms)
    min_ms = min(timings_ms)
    max_ms = max(timings_ms)
    return res, avg_ms, min_ms, max_ms, peak_mem_bytes


def testTemplate(customFunc, res_ref, is_fa_ref=False, warmup_iters=5, benchmark_iters=20, profile_once=True):
    res, avg_ms, min_ms, max_ms, peak_mem_bytes = benchmarkCudaOp(
        customFunc,
        warmup_iters=warmup_iters,
        benchmark_iters=benchmark_iters,
    )
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)
    print(
        f"cuda_time (avg/min/max over {benchmark_iters} iters) [ms]: "
        f"{avg_ms:.3f} ms / {min_ms:.3f} ms / {max_ms:.3f} ms"
    )
    print(f"cuda_peak_memory (max over {benchmark_iters} iters) [MB]: {peak_mem_mb:.2f} MB")
    if profile_once:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
            customFunc()
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    res_ref_cpu = res_ref.cpu().clone()
    res_cpu = res.cpu().clone()
    if is_fa_ref:
        res_cpu = res_cpu.transpose(1, 2)
    is_close = torch.allclose(res_ref_cpu, res_cpu, atol=1e-2, rtol=1e-4)
    max_abs_diff = (res_ref_cpu - res_cpu).abs().max().item()
    print(f"allclose={is_close}, max_abs_diff={max_abs_diff:.6f}")
    return {
        "avg_ms": avg_ms,
        "min_ms": min_ms,
        "max_ms": max_ms,
        "peak_mem_mb": peak_mem_mb,
        "allclose": is_close,
        "max_abs_diff": max_abs_diff,
    }


def mytest_simple():
    A = torch.tensor([1.0, 2.0, 3.0], device="cuda", dtype=torch.float16)
    B = torch.tensor([4.0, 5.0, 6.0], device="cuda", dtype=torch.float16)
    C = mr.mytest(A, B)
    
    expected = torch.tensor([5.0, 7.0, 9.0], device="cuda", dtype=torch.float16)
    assert torch.allclose(C, expected, atol=1e-3), f"Test failed! Expected {expected}, got {C}"
    print("Test passed! Result:", C)

def fa1Test(B, H, N, d, bc, br, running_times=5, use_wmma=False):
    print("Running Test: Flash Attention - 1\n")
    # shape1
    # N, d, B, H = 1024, 32, 1, 4
    Q,K,V = createQKVSimple(B, H, N, d)
    res_ref = badSoftmax(Q, K, V)
    Q_FA = Q.transpose(1, 2) # B, N, H, d
    K_FA = K.transpose(1, 2)
    V_FA = V.transpose(1, 2)
    attentionModuleStudent = CustomAttention(Q,K,V, None, None, None, B, H, N, d, False, bc, br, use_wmma=use_wmma)
    attentionModuleReference = CustomAttention(None, None, None, Q_FA, K_FA, V_FA, B, H, N, d, True, bc, br)
    ref_stats = []
    student_stats = []
    for i in range(running_times):
        print(f"-----RUNNING REFERENCE IMPLEMENTATION ({i})-----\n")
        ref_stats.append(
            testTemplate(
                attentionModuleReference.myFA1,
                res_ref,
                True,
                profile_once=(i == 0),
            )
        )
        time.sleep(3)
        print(f"-----RUNNING STUDENT IMPLEMENTATION ({i})-----\n")
        student_stats.append(
            testTemplate(
                attentionModuleStudent.myFA1,
                res_ref,
                profile_once=(i == 0),
            )
        )
        time.sleep(3)

    def summarize_stats(name, stats_list):
        avg_cuda_ms = sum(s["avg_ms"] for s in stats_list) / len(stats_list)
        avg_peak_mem_mb = sum(s["peak_mem_mb"] for s in stats_list) / len(stats_list)
        print(
            f"FINAL SUMMARY [{name}] -> "
            f"avg_cuda_time={avg_cuda_ms:.3f} ms, avg_peak_mem={avg_peak_mem_mb:.2f} MB"
        )

    print("\n===== FINAL BENCHMARK SUMMARY =====")
    summarize_stats("REFERENCE", ref_stats)
    summarize_stats("STUDENT", student_stats)


def fa1BackwardSmokeTest(B, H, N, d, bc, br, use_wmma=False):
    print("Running Test: Flash Attention - 1 Backward Smoke Test\n")
    Q, K, V = createQKVSimple(B, H, N, d)
    Q = Q.detach().requires_grad_(True)
    K = K.detach().requires_grad_(True)
    V = V.detach().requires_grad_(True)

    out = myFA1_autograd(Q, K, V, bc, br, use_wmma)
    loss = out.float().mean()
    loss.backward()

    for name, grad in (("Q", Q.grad), ("K", K.grad), ("V", V.grad)):
        has_grad = grad is not None
        all_finite = bool(torch.isfinite(grad).all().item()) if has_grad else False
        grad_norm = float(grad.float().norm().item()) if has_grad else float("nan")
        print(f"{name}.grad exists={has_grad}, finite={all_finite}, norm={grad_norm:.6f} (unitless)")


def main():

    d=32
    B=1
    H=4
    
    parser = argparse.ArgumentParser()
    parser.add_argument("testname", default="fa1", help="name of test to run: test, fa1, fa1_bw")
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    parser.add_argument("-bc",  default="32", help="Flash Attention Bc Size")
    parser.add_argument("-br", default="32", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Br Size")
    parser.add_argument("-d", default="32", help="Flash Attention head dimension")
    parser.add_argument("--impl", choices=["cuda", "wmma"], default="cuda", help="student kernel implementation")

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
        d = int(args.d)
        use_wmma = (args.impl == "wmma")
        if args.testname == "test":
            mytest_simple()
        elif args.testname == "fa1":
            # Keep argument order aligned with fa1Test(B, H, N, d, ...)
            print(
                f"fa1 config: B={B}, H={H}, N={N}, d={d}, "
                f"bc={int(args.bc)}, br={int(args.br)}, impl={args.impl}"
            )
            fa1Test(B, H, N, d, int(args.bc), int(args.br), use_wmma=use_wmma)
        elif args.testname == "fa1_bw":
            print(
                f"fa1_bw config: B={B}, H={H}, N={N}, d={d}, "
                f"bc={int(args.bc)}, br={int(args.br)}, impl={args.impl}"
            )
            fa1BackwardSmokeTest(B, H, N, d, int(args.bc), int(args.br), use_wmma=use_wmma)
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

        
if __name__ == "__main__":
    main()
