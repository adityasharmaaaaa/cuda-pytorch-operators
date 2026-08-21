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
    int64_t n=input.numel();
    int threads=256;
    int blocks=(n+threads-1)/threads;
    relu_forward_kernel<<<blocks,threads>>>(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        n
    );
    return output;
}
