import torch
from itertools import accumulate, chain

import extension_cpp

device = "cuda"
# dtype = torch.float32
dtype = torch.bfloat16
# dtype = torch.float16

int_dtype = torch.int64
# int_dtype = torch.int32

split_size_list = [4, 3, 1, 2]
num_repeats_list = [1, 2, 3, 5]
print(f"{split_size_list=}")
print(f"{num_repeats_list=}")

num_splits = len(split_size_list)
print(f"{num_splits=}")

cu_split_size_list = list(accumulate([0] + split_size_list))
print(f"{cu_split_size_list=}")

repeated_cu_split_size_list = list(accumulate(
    [0] + [split_size * num_repeats for split_size, num_repeats in zip(split_size_list, num_repeats_list)]
))
print(f"{repeated_cu_split_size_list=}")

seqlen = cu_split_size_list[-1]
print(f"{seqlen=}")

repeat_seqlen = repeated_cu_split_size_list[-1]
print(f"{repeat_seqlen=}")

nh, hd = 128, 1024
stride0 = nh * hd
print(f"{stride0=}")

recv_buffer = torch.zeros((seqlen, nh, hd), dtype=dtype, device=device)
print(f"Before range reduce: {recv_buffer=}")
repeated_recv_buffer = torch.ones((repeat_seqlen, nh, hd), dtype=dtype, device=device)

expected_recv_buffer = torch.tensor(list(chain(*[
    [num_repeats] * split_size * nh * hd
    for split_size, num_repeats in zip(split_size_list, num_repeats_list)
])), dtype=dtype, device=device).view(-1, nh, hd)

split_size_list = torch.tensor(split_size_list, dtype=int_dtype, device=device)
num_repeats_list = torch.tensor(num_repeats_list, dtype=int_dtype, device=device)
cu_split_size_list = torch.tensor(cu_split_size_list, dtype=int_dtype, device=device)
repeated_cu_split_size_list = torch.tensor(repeated_cu_split_size_list, dtype=int_dtype, device=device)

extension_cpp.ops.range_reduce(
    recv_buffer,
    repeated_recv_buffer,
    split_size_list,
    num_repeats_list,
    cu_split_size_list,
    repeated_cu_split_size_list,
    seqlen,
    num_splits,
    stride0,
)

torch.cuda.synchronize()

print(f"After range reduce: {recv_buffer=}")
print(f"Expected: {expected_recv_buffer=}")

assert torch.allclose(recv_buffer, expected_recv_buffer)