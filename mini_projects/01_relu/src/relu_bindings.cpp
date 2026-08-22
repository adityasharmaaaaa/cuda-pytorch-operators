#include <torch/extension.h>

torch::Tensor relu_forward_cuda(torch::Tensor input);
torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor output);

PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    m.def("relu_forward",&relu_forward_cuda,"Custom ReLU forward (CUDA)");
    m.def("relu_backward",&relu_backward_cuda, "Custom ReLU backward (CUDA)");
}
