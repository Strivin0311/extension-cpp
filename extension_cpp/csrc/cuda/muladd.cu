#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {

template <typename scalar_t>
__global__ void elementwise_add_kernel(
    scalar_t* output,
    const scalar_t* input,
    scalar_t value,
    int64_t num_elements
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_elements) {
        output[idx] = input[idx] + value;
    }
}

torch::Tensor add_scalar_cuda(torch::Tensor input, torch::Scalar value) {
    auto output = torch::empty_like(input);
    int64_t num_elements = input.numel();
    dim3 block(256);
    dim3 grid((num_elements + block.x - 1) / block.x);

    // 核心：类型分发宏
    // AT_DISPATCH_ALL_TYPES( /* 原始类型 */
    // AT_DISPATCH_ALL_TYPES_AND(at::ScalarType::BFloat16, /* 增加bfloat16一种类型 */
    AT_DISPATCH_ALL_TYPES_AND2(at::ScalarType::Half, at::ScalarType::BFloat16, /* 增加float16, bfloat16 两种类型*/
        input.scalar_type(), "elementwise_add", [&] {
            // 标量值转换为当前类型
            scalar_t val = value.to<scalar_t>(); /* scalar_t是宏里自动生成的类型，直接在lambda函数中使用 */
            elementwise_add_kernel<scalar_t><<<grid, block>>>(
                output.data_ptr<scalar_t>(),
                input.const_data_ptr<scalar_t>(),
                val,
                num_elements
            );
        }
    );

    return output;
}


__device__ int64_t binary_search_split(
    const int64_t* d_cu_split_size_list,
    int64_t start,
    int64_t end,
    int64_t idx,
    int64_t stride0
) {
    assert(start < end);
    int64_t low = start, high = end;
    while (low < high) { // [low, high)
        int64_t mid = low + (high - low) / 2;
        if (idx < d_cu_split_size_list[mid] * stride0) {
            high = mid; // [low, mid)
        } else {
            low = mid + 1; // [mid + 1, high)
        }
    }
    return low - 1; // low == high
}


template <typename scalar_t>
__global__ void range_reduce_kernel(
    scalar_t* recv_buffer,
    const scalar_t* repeated_recv_buffer,
    const int64_t* d_split_size_list,
    const int64_t* d_num_repeats_list,
    const int64_t* d_cu_split_size_list,
    const int64_t* d_repeated_cu_split_size_list,
    int64_t seqlen,
    int64_t num_splits,
    int64_t stride0
) {
    int64_t tid = blockDim.x * blockIdx.x + threadIdx.x;
    int64_t num_threads_per_grid = blockDim.x * gridDim.x;
    int64_t num_elements = seqlen * stride0;

    int64_t split_idx = 0;
    for (auto idx = tid; idx < num_elements; idx += num_threads_per_grid) {
        // search for split idx that the current idx belongs
        split_idx = binary_search_split(
            d_cu_split_size_list,
            split_idx,
            num_splits,
            idx,
            stride0
        );

        // get the info about this split
        auto recv_split_start = d_cu_split_size_list[split_idx] * stride0;
        auto recv_split_size = d_split_size_list[split_idx] * stride0;
        auto repeated_recv_split_start = d_repeated_cu_split_size_list[split_idx] * stride0;
        auto num_repeats = d_num_repeats_list[split_idx];
        auto recv_split_offset_to_idx = idx - recv_split_start;

        // load the recv data with its ptr that the current idx needs to reduce to
        scalar_t* recv_data_ptr = (recv_buffer + idx);
        scalar_t recv_reduce_data = *recv_data_ptr;

        // reduce the recv data from the corr. position in repeated_recv_buffer
        const scalar_t* repeated_recv_data_ptr = (repeated_recv_buffer + repeated_recv_split_start + recv_split_offset_to_idx);
        for (int64_t r = 0; r < num_repeats; ++r) {
            recv_reduce_data += *(repeated_recv_data_ptr + r * recv_split_size);
        }

        // write the reduced data back to recv_buffer
        *recv_data_ptr = recv_reduce_data;
    }
}

void range_reduce_cuda(
    at::Tensor& recv_buffer,
    at::Tensor& repeated_recv_buffer,
    at::Tensor& d_split_size_list,
    at::Tensor& d_num_repeats_list,
    at::Tensor& d_cu_split_size_list,
    at::Tensor& d_repeated_cu_split_size_list,
    int64_t seqlen,
    int64_t num_splits,
    int64_t stride0
) {
    dim3 gridDims(10); dim3 blockDims(512);
    AT_DISPATCH_ALL_TYPES_AND2(
      at::ScalarType::Half, at::ScalarType::BFloat16,
      recv_buffer.scalar_type(), 
      "group_reduce_nccl_post_process", 
      [&] {
          range_reduce_kernel<scalar_t>
          <<<gridDims, blockDims>>>(
              recv_buffer.data_ptr<scalar_t>(),
              repeated_recv_buffer.data_ptr<scalar_t>(),
              d_split_size_list.data_ptr<int64_t>(),
              d_num_repeats_list.data_ptr<int64_t>(),
              d_cu_split_size_list.data_ptr<int64_t>(),
              d_repeated_cu_split_size_list.data_ptr<int64_t>(),
              seqlen,
              num_splits,
              stride0
          );
        }
    );
}


__global__ void muladd_kernel(int numel, const float* a, const float* b, float c, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx] + c;
}

at::Tensor mymuladd_cuda(const at::Tensor& a, const at::Tensor& b, double c) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());

  // test c10::IntArrayRef
  // auto a_size = a.sizes();
  // std::vector<int64_t> a_shape(a_size.begin(), a_size.end());
  // printf("original a shape: ");
  // for (auto val : a_shape) {
  //   printf("%lld ", val);
  // }
  // printf("\n");

  // a_shape[0] *= 2;
  // // auto a_shape_ext = c10::IntArrayRef(a_shape);
  // auto a_shape_ext = c10::makeArrayRef(a_shape);
  // printf("modified a shape (first dim doubled): ");
  // for (auto val : a_shape_ext) {
  //   printf("%lld ", val);
  // }
  // printf("\n");

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();

  int numel = a_contig.numel();
  muladd_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, c, result_ptr);
  return result;
}

__global__ void mul_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] * b[idx];
}

at::Tensor mymul_cuda(const at::Tensor& a, const at::Tensor& b) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  at::Tensor result = at::empty(a_contig.sizes(), a_contig.options());

  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = result.data_ptr<float>();
  int numel = a_contig.numel();
  mul_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
  return result;
}

__global__ void add_kernel(int numel, const float* a, const float* b, float* result) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < numel) result[idx] = a[idx] + b[idx];
}

void myadd_out_cuda(const at::Tensor& a, const at::Tensor& b, at::Tensor& out) {
  TORCH_CHECK(a.sizes() == b.sizes());
  TORCH_CHECK(b.sizes() == out.sizes());
  TORCH_CHECK(a.dtype() == at::kFloat);
  TORCH_CHECK(b.dtype() == at::kFloat);
  TORCH_CHECK(out.dtype() == at::kFloat);
  TORCH_CHECK(out.is_contiguous());
  TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
  TORCH_INTERNAL_ASSERT(out.device().type() == at::DeviceType::CUDA);

  at::Tensor a_contig = a.contiguous();
  at::Tensor b_contig = b.contiguous();
  const float* a_ptr = a_contig.data_ptr<float>();
  const float* b_ptr = b_contig.data_ptr<float>();
  float* result_ptr = out.data_ptr<float>();
  int numel = a_contig.numel();
  add_kernel<<<(numel+255)/256, 256>>>(numel, a_ptr, b_ptr, result_ptr);
}

// Registers CUDA implementations for mymuladd, mymul, myadd_out
TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
  m.impl("mymuladd", &mymuladd_cuda);
  m.impl("mymul", &mymul_cuda);
  m.impl("myadd_out", &myadd_out_cuda);
  m.impl("add_scalar", &add_scalar_cuda);
  m.impl("range_reduce", &range_reduce_cuda);
}

}
