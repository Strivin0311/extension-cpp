// #include <ATen/Operators.h>

// #include <torch/all.h>
// #include <torch/library.h>

// #include <vector>


// namespace extension_cpp {

//     void mulsub_func(const float* a, const float* b, float* c, double s, int numel) {
//         for (auto i = 0; i < numel; i++) {
//             c[i] = a[i] * b[i] - s;
//         }
//     }

//     at::Tensor mymulsub_cpu(const at::Tensor& a, const at::Tensor& b, double s) {
//         // check input tensors
//         TORCH_CHECK(a.sizes() == b.sizes());
//         TORCH_CHECK(a.dtype() == at::kFloat);
//         TORCH_CHECK(b.dtype() == at::kFloat);
//         TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);
//         TORCH_INTERNAL_ASSERT(a.device().type() == at::DeviceType::CPU);

//         // make input tensors contiguous
//         at::Tensor a_contig = a.contiguous();
//         at::Tensor b_contig = b.contiguous();
        
//         // allocate output tensor
//         at::Tensor c = at::empty(a_contig.sizes(), a_contig.options());

//         // get pointers
//         const float* a_ptr = a_contig.data_ptr<float>();
//         const float* b_ptr = b_contig.data_ptr<float>();
//         float* c_ptr = c.data_ptr<float>();

//         // make meta args
//         int numel = a_contig.numel();

//         // launch cpu func
//         mulsub_func(a_ptr, b_ptr, c_ptr, s, numel);

//         return c;
//     }

//     // define the operator to torch library
//     TORCH_LIBRARY(extension_cpp, m) {
//         m.def("mymulsub(Tensor a, Tensor b, float s) -> Tensor");
//     }

//     // register the cpu implementation of the operator to torch library
//     TORCH_LIBRARY_IMPL(extension_cpp, CPU, m) {
//         m.impl("mymulsub", &mymulsub_cpu);
//     }

// } // namespace extension_cpp