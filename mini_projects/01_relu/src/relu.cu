#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

template <typename scalar_t>
__global__ void relu_forward_kernel(const scalar_t* input, scalar_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        output[idx] = input[idx] > scalar_t(0) ? input[idx] : scalar_t(0);
    }
}

template <typename scalar_t>
__global__ void relu_backward_kernel(const scalar_t* grad_output, scalar_t* grad_input,
                                      const scalar_t* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        grad_input[idx] = output[idx] > scalar_t(0) ? grad_output[idx] : scalar_t(0);
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor input) {
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    auto output = torch::empty_like(input);
    int n = input.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(input.scalar_type(), "relu_forward_cuda", ([&] {
        relu_forward_kernel<scalar_t><<<blocks, threads>>>(
            input.data_ptr<scalar_t>(), output.data_ptr<scalar_t>(), n);
    }));
    return output;
}

torch::Tensor relu_backward_cuda(torch::Tensor grad_output, torch::Tensor output) {
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.numel() == output.numel(), "grad_output and output must have the same number of elements");
    auto grad_input = torch::empty_like(grad_output);
    int n = output.numel();
    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    AT_DISPATCH_FLOATING_TYPES(output.scalar_type(), "relu_backward_cuda", ([&] {
        relu_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(), grad_input.data_ptr<scalar_t>(),
            output.data_ptr<scalar_t>(), n);
    }));
    return grad_input;
}