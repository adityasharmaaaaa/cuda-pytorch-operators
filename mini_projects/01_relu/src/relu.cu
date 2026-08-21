#include <torch/extension.h>
#include <cuda_runtime.h>

__global__ void relu_forward_kernel(const float* input, float* output, int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        float val=input[idx];
        output[idx]=isnan(val)?val:fmaxf(0.0f, val);
    }
}

torch::Tensor relu_forward_cuda(torch::Tensor input){
    TORCH_CHECK(input.is_cuda(), "input must be a CUDA tensor");
    TORCH_CHECK(input.is_contiguous(), "input must be contiguous");
    TORCH_CHECK(input.scalar_type()==torch::kFloat32, "input must be float32");
    auto output=torch::empty_like(input);
    int n=input.numel();
    int threads=256;
    int blocks=(n+threads-1)/threads;
    relu_forward_kernel<<<blocks,threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output;
}


__global__ void relu_backward_kernel(const float* grad_output,float* grad_input, const float* output, int N){
    int idx=blockIdx.x*blockDim.x+threadIdx.x;
    if(idx<N){
        grad_input[idx]=(output[idx]>0.0f)?grad_output[idx]:0.0f;
    }
}

torch::Tensor relu_backward_cuda(torch:: Tensor grad_output, torch::Tensor output){
    TORCH_CHECK(output.is_cuda(), "output must be a CUDA tensor");
    TORCH_CHECK(output.is_contiguous(), "output must be contiguous");
    TORCH_CHECK(output.scalar_type()==torch::kFloat32, "output must be float32");
    TORCH_CHECK(grad_output.is_cuda(), "grad_output must be a CUDA tensor");
    TORCH_CHECK(grad_output.is_contiguous(), "grad_output must be contiguous");
    TORCH_CHECK(grad_output.scalar_type()==torch::kFloat32, "grad_output must be float32");
    TORCH_CHECK(grad_output.numel() == output.numel(), "grad_output and output must have the same number of elements");
    auto grad_input=torch::empty_like(grad_output);
    int n=output.numel();
    int threads=256;
    int blocks=(n+threads-1)/threads;
    relu_backward_kernel<<<blocks,threads>>>(
        grad_output.data_ptr<float>(),
        grad_input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return grad_input;
}