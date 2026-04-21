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
    sources=["module.cpp", "kernel.cu", "kernel_fa2.cu", "kernel_fa2_bwd.cu"],
    extra_cuda_cflags=["-arch=sm_120"],
    build_directory='./build',
    verbose=False
)


class MyFA2Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, Q, K, V, bc, br, causal: bool):
        B, H, N, d = Q.shape
        L = torch.zeros((B, H, N), device=Q.device, dtype=torch.float32)
        M = torch.zeros((B, H, N), device=Q.device, dtype=torch.float32)
        O = mr.myFA2(
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
            int(causal),
        )
        ctx.save_for_backward(Q, K, V, L)
        ctx.bc = int(bc)
        ctx.br = int(br)
        ctx.causal = bool(causal)
        return O

    @staticmethod
    def backward(ctx, dO):
        Q, K, V, L = ctx.saved_tensors
        bc = ctx.bc
        br = ctx.br
        causal = ctx.causal
        B, H, N, d = Q.shape
        dQ = torch.zeros_like(Q)
        dK = torch.zeros_like(K)
        dV = torch.zeros_like(V)
        mr.myFA2_backward(
            dQ,
            dK,
            dV,
            Q.contiguous(),
            K.contiguous(),
            V.contiguous(),
            dO.contiguous(),
            L,
            bc,
            br,
            B,
            H,
            N,
            d,
            int(causal),
        )
        return dQ, dK, dV, None, None, None


def myFA2_autograd(Q, K, V, bc, br, causal: bool = False):
    return MyFA2Function.apply(Q, K, V, bc, br, causal)


class CustomAttention(nn.Module):
    def __init__(
        self,
        Q,
        K,
        V,
        Q_FA,
        K_FA,
        V_FA,
        B,
        H,
        N,
        d,
        isRef=False,
        bc=256,
        br=256,
        causal=False,
    ):
        super(nn.Module, self).__init__()
        self.Q = Q
        self.K = K
        self.V = V
        self.Q_FA = Q_FA
        self.K_FA = K_FA
        self.V_FA = V_FA
        self.B = B
        self.H = H
        self.N = N
        self.d = d
        self.isRef = isRef
        self.bc = bc
        self.br = br
        self.causal = causal

    def run_forward(self):
        if self.Q is not None:
            device = self.Q.device
        else:
            device = self.Q_FA.device
        d = self.d
        if self.isRef:
            with record_function("REFERENCE - FLASH ATTENTION"):
                out = flash_attn_func(
                    self.Q_FA, self.K_FA, self.V_FA, causal=self.causal
                )
            return out
        Q = self.Q.contiguous()
        K = self.K.contiguous()
        V = self.V.contiguous()
        with record_function("STUDENT - FLASH ATTENTION - v2"):
            out = myFA2_autograd(Q, K, V, self.bc, self.br, self.causal)
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

def badSoftmax(Q, K, V, causal: bool = False):
    # 输入形状为 (B, H, N, d)
    d = Q.shape[-1]
    QK = Q @ K.transpose(-2, -1) * (1.0 / math.sqrt(d))
    if causal:
        N = QK.shape[-1]
        mask = torch.triu(
            torch.ones((N, N), device=QK.device, dtype=torch.bool), diagonal=1
        )
        QK = QK.masked_fill(mask.view(1, 1, N, N), float("-inf"))

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


def benchmarkBackwardOp(customFunc, warmup_iters=5, benchmark_iters=20):
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

def flash_attn_benchmark(B, H, N, d, bc, br, running_times=5, causal=False):
    print("Running Test: FlashAttention-2 forward (student CUDA vs reference)\n")
    Q, K, V = createQKVSimple(B, H, N, d)
    res_ref = badSoftmax(Q, K, V, causal=causal)
    Q_FA = Q.transpose(1, 2)  # B, N, H, d
    K_FA = K.transpose(1, 2)
    V_FA = V.transpose(1, 2)
    attention_module_student = CustomAttention(
        Q, K, V, None, None, None, B, H, N, d, False, bc, br, causal
    )
    attention_module_reference = CustomAttention(
        None, None, None, Q_FA, K_FA, V_FA, B, H, N, d, True, bc, br, causal
    )
    ref_stats = []
    student_stats = []
    for i in range(running_times):
        print(f"-----RUNNING REFERENCE IMPLEMENTATION ({i})-----\n")
        ref_stats.append(
            testTemplate(
                attention_module_reference.run_forward,
                res_ref,
                True,
                profile_once=(i == 0),
            )
        )
        time.sleep(3)
        print(f"-----RUNNING STUDENT IMPLEMENTATION ({i})-----\n")
        student_stats.append(
            testTemplate(
                attention_module_student.run_forward,
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


def fa2Test(B, H, N, d, bc, br, running_times=5, causal=False):
    flash_attn_benchmark(B, H, N, d, bc, br, running_times, causal=causal)


def fa2_backward_smoke_test(B, H, N, d, bc, br, causal=False):
    print("Running Test: Flash Attention - 2 backward benchmark (CUDA vs PyTorch reference)\n")
    torch.manual_seed(0)
    Q0 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    K0 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)
    V0 = torch.randn(B, H, N, d, device="cuda", dtype=torch.float16)

    def student_backward_once():
        Q = Q0.detach().clone().requires_grad_(True)
        K = K0.detach().clone().requires_grad_(True)
        V = V0.detach().clone().requires_grad_(True)
        out = myFA2_autograd(Q, K, V, bc, br, causal)
        out.float().sum().backward()
        return Q.grad.detach(), K.grad.detach(), V.grad.detach()

    def reference_backward_once():
        Qr = Q0.detach().clone().float().requires_grad_(True)
        Kr = K0.detach().clone().float().requires_grad_(True)
        Vr = V0.detach().clone().float().requires_grad_(True)
        scale = 1.0 / math.sqrt(d)
        scores = torch.matmul(Qr, Kr.transpose(-2, -1)) * scale
        if causal:
            mask = torch.triu(
                torch.ones((N, N), device=scores.device, dtype=torch.bool), diagonal=1
            )
            scores = scores.masked_fill(mask.view(1, 1, N, N), float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        out_ref = torch.matmul(probs, Vr)
        out_ref.sum().backward()
        return Qr.grad.detach(), Kr.grad.detach(), Vr.grad.detach()

    print("-----RUNNING REFERENCE BACKWARD-----\n")
    ref_grads, ref_avg_ms, ref_min_ms, ref_max_ms, ref_peak_bytes = benchmarkBackwardOp(
        reference_backward_once, 5, 20
    )
    print(
        f"reference backward cuda_time (avg/min/max over 20 iters) [ms]: "
        f"{ref_avg_ms:.3f} / {ref_min_ms:.3f} / {ref_max_ms:.3f}"
    )
    print(f"reference backward peak memory [MB]: {ref_peak_bytes / (1024*1024):.2f}")

    print("-----RUNNING STUDENT BACKWARD-----\n")
    stu_grads, stu_avg_ms, stu_min_ms, stu_max_ms, stu_peak_bytes = benchmarkBackwardOp(
        student_backward_once, 5, 20
    )
    print(
        f"student backward cuda_time (avg/min/max over 20 iters) [ms]: "
        f"{stu_avg_ms:.3f} / {stu_min_ms:.3f} / {stu_max_ms:.3f}"
    )
    print(f"student backward peak memory [MB]: {stu_peak_bytes / (1024*1024):.2f}")

    atol, rtol = 5e-2, 1e-2
    names = ("Q", "K", "V")
    for name, g_cuda, g_ref in zip(names, stu_grads, ref_grads):
        ok = torch.allclose(g_cuda.float(), g_ref, atol=atol, rtol=rtol)
        mad = (g_cuda.float() - g_ref).abs().max().item()
        print(f"{name}.grad allclose={ok}, max_abs_diff={mad:.6f} (atol={atol}, rtol={rtol})")


def main():

    d=32
    B=1
    H=4
    
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "testname",
        default="fa2",
        help="name of test to run: test, fa2, fa2_bw",
    )
    parser.add_argument("-m", "--model", default="shakes128", help="name of model to use: shakes128, shakes1024, shakes2048, kayvon")
    parser.add_argument("--inference", action="store_true", default=False, help="run gpt inference")
    parser.add_argument("-bc",  default="32", help="Flash Attention Bc Size")
    parser.add_argument("-br", default="32", help="Flash Attention Br Size")
    parser.add_argument("-N", default="1024", help="Flash Attention Br Size")
    parser.add_argument("-d", default="32", help="Flash Attention head dimension")
    parser.add_argument(
        "--causal",
        action="store_true",
        help="Use causal masking (forward + backward)",
    )
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
        if args.testname == "test":
            mytest_simple()
        elif args.testname == "fa2":
            print(
                f"fa2 config: B={B}, H={H}, N={N}, d={d}, "
                f"bc={int(args.bc)}, br={int(args.br)}, causal={args.causal} "
                f"(FlashAttention-2 student kernel)"
            )
            fa2Test(B, H, N, d, int(args.bc), int(args.br), causal=args.causal)
        elif args.testname == "fa2_bw":
            print(
                f"fa2_bw config: B={B}, H={H}, N={N}, d={d}, "
                f"bc={int(args.bc)}, br={int(args.br)}, causal={args.causal}"
            )
            fa2_backward_smoke_test(B, H, N, d, int(args.bc), int(args.br), causal=args.causal)
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

        
if __name__ == "__main__":
    main()
