import torch
from torch.profiler import profile, record_function, ProfilerActivity, schedule

def trace_handler(p):
    print(p.key_averages(group_by_input_shape=True).table(
        sort_by="self_cuda_time_total", row_limit=10
    ))

with profile(
    activities=[ProfilerActivity.CUDA],
    profile_memory=True,
    record_shapes=True,
    with_modules=True,          # ✅ 加上这一行
    schedule=schedule(wait=1, warmup=1, active=3),
    on_trace_ready=trace_handler
) as prof:
    for step in range(5):
        with record_function("matmul"):
            a = torch.randn(1000, 1000).cuda()
            b = torch.randn(1000, 1000).cuda()
            c = a @ b
        prof.step()

# with torch.autograd.profiler.profile(use_device='cuda') as prof:
#     torch.randn(1000, 1000).cuda() @ torch.randn(1000, 1000).cuda()
# print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=10))