#include <cuda.h>
#include <cuda_runtime.h>
#include <torch/extension.h>

__global__ void square_kernel (float* x, float* out, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        out[idx] = x[idx] * x[idx];
    }
}

void launch_square (torch::Tensor input, torch::Tensor output) {
    int size    = input.numel ();
    int threads = 256;
    int blocks  = (size + threads - 1) / threads;

    square_kernel<<<blocks, threads>>> (
    input.data_ptr<float> (), output.data_ptr<float> (), size);
}
