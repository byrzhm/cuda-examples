__global__ void
add_residual_kernel (const float* x, const float* residual, float* out, int n) {
    // implement elementwise residual addition out[i] = x[i] + residual[i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        out[idx] = x[idx] + residual[idx];
}
