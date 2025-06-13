#include <ATen/Operators.h>
#include <torch/all.h>
#include <torch/library.h>

#include <cuda.h>
#include <cuda_runtime.h>

namespace extension_cpp {
    __global__ void mulsub_kernel(const float* a, const float* b, float* c, float s, int numel) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < numel) {
            c[idx] = a[idx] * b[idx] - s;
        }
    }

    at::Tensor mymulsub_cuda(const at::Tensor& a, const at::Tensor& b, double s) {
        // check input tensors
        TORCH_CHECK(a.sizes() == b.sizes());
        TORCH_CHECK(a.dtype() == at::kFloat);
        TORCH_CHECK(b.dtype() == at::kFloat);
        TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CUDA);
        TORCH_INTERNAL_ASSERT(b.device().type() == at::DeviceType::CUDA);
        
        // make input tensors contiguous
        at::Tensor a_contig = a.contiguous();
        at::Tensor b_contig = b.contiguous();

        // allocate output tensor
        at::Tensor c = at::empty(a_contig.sizes(), a_contig.options());

        // get pointers
        const float* a_ptr = a_contig.data_ptr<float>();
        const float* b_ptr = b_contig.data_ptr<float>();
        float* c_ptr = c.data_ptr<float>();

        // make meta args of the kernel
        int numel = a_contig.numel();
        int block_size = 256;
        int num_blocks = (numel + block_size - 1) / block_size;
        dim3 blockDims(block_size); dim3 gridDims(num_blocks);

        // launch kernel
        mulsub_kernel<<<gridDims, blockDims>>>(a_ptr, b_ptr, c_ptr, s, numel);

        return c;
    }

    // register the cuda implementation of the operator to torch library
    TORCH_LIBRARY_IMPL(extension_cpp, CUDA, m) {
        m.impl("mymulsub", &mymulsub_cuda);
    }
    
} // namespace extension_cpp