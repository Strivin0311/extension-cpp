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
            scalar_t val = value.to<scalar_t>();
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
}

}
