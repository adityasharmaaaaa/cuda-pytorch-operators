#include <torch/extension.h>

torch::Tensor relu_forward_cuda(torch::Tensor input);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("relu_forward",&relu_forward_cuda,"Custom ReLU forward (CUDA)");
}