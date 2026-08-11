__global__ void silu_kernel (const float* x, float* out, int n) {
    // apply SiLU elementwise: out[i] = x[i] / (1 + exp(-x[i]))
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_val = x[idx];
        out[idx] = x_val / (1.0f + expf(-x_val));
    }
}
