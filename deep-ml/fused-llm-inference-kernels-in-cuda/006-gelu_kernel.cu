__global__ void gelu_kernel(const float* x, float* out, int n) {
    // Apply GELU (tanh approximation) elementwise to x, write into out
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float x_val = x[idx];
        float cdf = 0.5f * (1.0f + tanhf(0.7978845608028654f * (x_val + 0.044715f * x_val * x_val * x_val)));
        out[idx] = x_val * cdf;
    }
}
