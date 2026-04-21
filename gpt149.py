import argparse
import time
import math
import random
import inspect
import json
import os
from dataclasses import dataclass
import sys, getopt
from os import getcwd, path

# Keep benchmarking in eager mode; avoid accidental torch.compile/dynamo paths.
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.cpp_extension import load
from torch.profiler import profile, record_function, ProfilerActivity
from flash_attn import flash_attn_func
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


def maybe_profile_cuda_region(work_fn, enabled=False, label="region", iters=1):
    if not enabled:
        return
    cudart = torch.cuda.cudart()
    torch.cuda.synchronize()
    cudart.cudaProfilerStart()
    for i in range(iters):
        torch.cuda.nvtx.range_push(f"{label}_iter_{i}")
        work_fn()
        torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    cudart.cudaProfilerStop()


def print_bench_line(tag, avg_ms, min_ms, max_ms, peak_mem_bytes):
    print(
        f"{tag:>18s} | "
        f"time avg/min/max [ms] = {avg_ms:.3f}/{min_ms:.3f}/{max_ms:.3f} | "
        f"peak mem [MB] = {peak_mem_bytes / (1024 * 1024):.2f}"
    )


def compare_forward_to_ref(name, out, ref):
    out_cpu = out.detach().cpu().float()
    ref_cpu = ref.detach().cpu().float()
    ok = torch.allclose(out_cpu, ref_cpu, atol=1e-2, rtol=1e-4)
    mad = (out_cpu - ref_cpu).abs().max().item()
    print(f"{name:>18s} | allclose={ok}, max_abs_diff={mad:.6f}")


def compare_grads_to_ref(name, grads, ref_grads, atol=5e-2, rtol=1e-2):
    for gname, g, rg in zip(("Q", "K", "V"), grads, ref_grads):
        ok = torch.allclose(g.float(), rg.float(), atol=atol, rtol=rtol)
        mad = (g.float() - rg.float()).abs().max().item()
        print(
            f"{name:>18s} | {gname}.grad allclose={ok}, "
            f"max_abs_diff={mad:.6f} (atol={atol}, rtol={rtol})"
        )


def summarize_stats(stats):
    avg_t = sum(x[0] for x in stats) / len(stats)
    avg_m = sum(x[1] for x in stats) / len(stats) / (1024 * 1024)
    return {"avg_ms": avg_t, "avg_peak_mem_mb": avg_m}


def maybe_write_json(payload, json_out):
    if not json_out:
        return
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[json] wrote benchmark summary -> {json_out}")


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

def flash_attn_benchmark(
    B,
    H,
    N,
    d,
    bc,
    br,
    running_times=5,
    causal=False,
    profile_ncu=False,
    profile_iters=1,
    json_out="",
):
    print("Running Test: FlashAttention forward benchmark (torch vs flash-attn vs student)\n")
    Q, K, V = createQKVSimple(B, H, N, d)
    Q_FA = Q.transpose(1, 2).contiguous()  # B, N, H, d
    K_FA = K.transpose(1, 2).contiguous()
    V_FA = V.transpose(1, 2).contiguous()

    def torch_manual_forward():
        with torch.no_grad():
            return badSoftmax(Q, K, V, causal=causal)

    def flash_lib_forward():
        with torch.no_grad():
            return flash_attn_func(Q_FA, K_FA, V_FA, causal=causal).transpose(1, 2)

    def student_forward():
        with torch.no_grad():
            return myFA2_autograd(Q, K, V, bc, br, causal)

    torch_stats = []
    flash_stats = []
    student_stats = []
    last_cmp = {}
    ref_out = torch_manual_forward()

    for i in range(running_times):
        print(f"-----FORWARD RUN ({i})-----\n")
        out_torch, t_avg, t_min, t_max, t_mem = benchmarkCudaOp(torch_manual_forward)
        out_flash, f_avg, f_min, f_max, f_mem = benchmarkCudaOp(flash_lib_forward)
        out_student, s_avg, s_min, s_max, s_mem = benchmarkCudaOp(student_forward)

        print_bench_line("torch_manual", t_avg, t_min, t_max, t_mem)
        print_bench_line("flash_attn_lib", f_avg, f_min, f_max, f_mem)
        print_bench_line("student_cuda", s_avg, s_min, s_max, s_mem)

        compare_forward_to_ref("torch_manual", out_torch, ref_out)
        compare_forward_to_ref("flash_attn_lib", out_flash, ref_out)
        compare_forward_to_ref("student_cuda", out_student, ref_out)
        last_cmp = {
            "torch_manual_max_abs_diff": float((out_torch.detach().float() - ref_out.detach().float()).abs().max().item()),
            "flash_attn_max_abs_diff": float((out_flash.detach().float() - ref_out.detach().float()).abs().max().item()),
            "student_max_abs_diff": float((out_student.detach().float() - ref_out.detach().float()).abs().max().item()),
        }

        torch_stats.append((t_avg, t_mem))
        flash_stats.append((f_avg, f_mem))
        student_stats.append((s_avg, s_mem))

        if profile_ncu and i == 0:
            print("\n[profiler] Capturing one region per implementation for Nsight Compute")
            maybe_profile_cuda_region(torch_manual_forward, True, "fwd_torch_manual", profile_iters)
            maybe_profile_cuda_region(flash_lib_forward, True, "fwd_flash_lib", profile_iters)
            maybe_profile_cuda_region(student_forward, True, "fwd_student", profile_iters)
        time.sleep(3)

    print("\n===== FINAL BENCHMARK SUMMARY =====")
    summary = {}
    for name, stats in (
        ("torch_manual", torch_stats),
        ("flash_attn_lib", flash_stats),
        ("student_cuda", student_stats),
    ):
        s = summarize_stats(stats)
        summary[name] = s
        print(f"{name:>18s} | avg cuda time [ms] = {s['avg_ms']:.3f}, avg peak mem [MB] = {s['avg_peak_mem_mb']:.2f}")

    payload = {
        "test": "fa2_forward_benchmark",
        "config": {"B": B, "H": H, "N": N, "d": d, "bc": bc, "br": br, "causal": bool(causal), "runs": running_times},
        "summary": summary,
        "last_run_max_abs_diff": last_cmp,
    }
    maybe_write_json(payload, json_out)
    return payload


def fa2Test(B, H, N, d, bc, br, running_times=5, causal=False, profile_ncu=False, profile_iters=1, json_out=""):
    return flash_attn_benchmark(
        B,
        H,
        N,
        d,
        bc,
        br,
        running_times,
        causal=causal,
        profile_ncu=profile_ncu,
        profile_iters=profile_iters,
        json_out=json_out,
    )


def fa2_backward_benchmark(B, H, N, d, bc, br, causal=False, profile_ncu=False, profile_iters=1, json_out=""):
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

    def flash_backward_once():
        Qf = Q0.detach().clone().transpose(1, 2).contiguous().requires_grad_(True)
        Kf = K0.detach().clone().transpose(1, 2).contiguous().requires_grad_(True)
        Vf = V0.detach().clone().transpose(1, 2).contiguous().requires_grad_(True)
        out = flash_attn_func(Qf, Kf, Vf, causal=causal)
        out.float().sum().backward()
        return (
            Qf.grad.detach().transpose(1, 2).contiguous(),
            Kf.grad.detach().transpose(1, 2).contiguous(),
            Vf.grad.detach().transpose(1, 2).contiguous(),
        )

    # Kernel-only student backward (no autograd graph construction overhead).
    L_kernel = torch.zeros((B, H, N), device="cuda", dtype=torch.float32)
    M_kernel = torch.zeros((B, H, N), device="cuda", dtype=torch.float32)
    _ = mr.myFA2(
        Q0.contiguous(),
        K0.contiguous(),
        V0.contiguous(),
        L_kernel,
        M_kernel,
        int(bc),
        int(br),
        B,
        H,
        N,
        d,
        int(causal),
    )
    dO_kernel = torch.ones((B, H, N, d), device="cuda", dtype=torch.float16)

    def student_backward_kernel_only_once():
        dQ = torch.zeros_like(Q0)
        dK = torch.zeros_like(K0)
        dV = torch.zeros_like(V0)
        mr.myFA2_backward(
            dQ,
            dK,
            dV,
            Q0.contiguous(),
            K0.contiguous(),
            V0.contiguous(),
            dO_kernel.contiguous(),
            L_kernel,
            int(bc),
            int(br),
            B,
            H,
            N,
            d,
            int(causal),
        )
        return dQ, dK, dV

    print("-----RUNNING BACKWARD BENCHMARKS-----\n")
    ref_grads, ref_avg_ms, ref_min_ms, ref_max_ms, ref_peak_bytes = benchmarkBackwardOp(reference_backward_once, 5, 20)
    flash_grads, fla_avg_ms, fla_min_ms, fla_max_ms, fla_peak_bytes = benchmarkBackwardOp(flash_backward_once, 5, 20)
    stu_grads, stu_avg_ms, stu_min_ms, stu_max_ms, stu_peak_bytes = benchmarkBackwardOp(student_backward_once, 5, 20)
    _, stu_k_avg_ms, stu_k_min_ms, stu_k_max_ms, stu_k_peak_bytes = benchmarkBackwardOp(
        student_backward_kernel_only_once, 5, 20
    )

    print_bench_line("torch_manual_bw", ref_avg_ms, ref_min_ms, ref_max_ms, ref_peak_bytes)
    print_bench_line("flash_attn_bw", fla_avg_ms, fla_min_ms, fla_max_ms, fla_peak_bytes)
    print_bench_line("student_bw", stu_avg_ms, stu_min_ms, stu_max_ms, stu_peak_bytes)
    print_bench_line(
        "student_bw_kernel",
        stu_k_avg_ms,
        stu_k_min_ms,
        stu_k_max_ms,
        stu_k_peak_bytes,
    )

    print("   torch_manual_bw | baseline reference gradients")
    compare_grads_to_ref("flash_attn_bw", flash_grads, ref_grads)
    compare_grads_to_ref("student_bw", stu_grads, ref_grads)

    atol, rtol = 5e-2, 1e-2
    grad_cmp = {}
    for tag, grads in (("flash_attn_bw", flash_grads), ("student_bw", stu_grads)):
        grad_cmp[tag] = {}
        for gname, g, rg in zip(("Q", "K", "V"), grads, ref_grads):
            grad_cmp[tag][gname] = {
                "allclose": bool(torch.allclose(g.float(), rg.float(), atol=atol, rtol=rtol)),
                "max_abs_diff": float((g.float() - rg.float()).abs().max().item()),
            }

    if profile_ncu:
        print("\n[profiler] Capturing one backward region per implementation for Nsight Compute")
        maybe_profile_cuda_region(reference_backward_once, True, "bwd_torch_manual", profile_iters)
        maybe_profile_cuda_region(flash_backward_once, True, "bwd_flash_lib", profile_iters)
        maybe_profile_cuda_region(student_backward_once, True, "bwd_student", profile_iters)

    payload = {
        "test": "fa2_backward_benchmark",
        "config": {"B": B, "H": H, "N": N, "d": d, "bc": bc, "br": br, "causal": bool(causal)},
        "timing": {
            "torch_manual_bw": {"avg_ms": ref_avg_ms, "min_ms": ref_min_ms, "max_ms": ref_max_ms, "peak_mem_mb": ref_peak_bytes / (1024 * 1024)},
            "flash_attn_bw": {"avg_ms": fla_avg_ms, "min_ms": fla_min_ms, "max_ms": fla_max_ms, "peak_mem_mb": fla_peak_bytes / (1024 * 1024)},
            "student_bw": {"avg_ms": stu_avg_ms, "min_ms": stu_min_ms, "max_ms": stu_max_ms, "peak_mem_mb": stu_peak_bytes / (1024 * 1024)},
            "student_bw_kernel": {"avg_ms": stu_k_avg_ms, "min_ms": stu_k_min_ms, "max_ms": stu_k_max_ms, "peak_mem_mb": stu_k_peak_bytes / (1024 * 1024)},
        },
        "grad_compare_to_torch_manual": grad_cmp,
    }
    maybe_write_json(payload, json_out)
    return payload


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
    parser.add_argument(
        "--profile-ncu",
        action="store_true",
        help="Enable cudaProfilerStart/Stop + NVTX ranges for Nsight Compute capture",
    )
    parser.add_argument(
        "--profile-iters",
        default="1",
        help="Iterations captured inside profiler region when --profile-ncu is set",
    )
    parser.add_argument(
        "--json-out",
        default="",
        help="Optional path to write benchmark summary JSON",
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
            fa2Test(
                B,
                H,
                N,
                d,
                int(args.bc),
                int(args.br),
                causal=args.causal,
                profile_ncu=args.profile_ncu,
                profile_iters=int(args.profile_iters),
                json_out=args.json_out,
            )
        elif args.testname == "fa2_bw":
            print(
                f"fa2_bw config: B={B}, H={H}, N={N}, d={d}, "
                f"bc={int(args.bc)}, br={int(args.br)}, causal={args.causal}"
            )
            fa2_backward_benchmark(
                B,
                H,
                N,
                d,
                int(args.bc),
                int(args.br),
                causal=args.causal,
                profile_ncu=args.profile_ncu,
                profile_iters=int(args.profile_iters),
                json_out=args.json_out,
            )
        else:
            print("Unknown test name: %s" % args.testname)
    else:
        print("Running inference using dnn model %s" % (args.model))
        from sample import run_sample
        run_sample(N, model_filename, args.testname)

        
if __name__ == "__main__":
    main()
