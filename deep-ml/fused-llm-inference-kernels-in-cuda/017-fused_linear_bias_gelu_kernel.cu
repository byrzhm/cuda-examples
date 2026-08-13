__global__ void fused_linear_bias_gelu_kernel(
    const float* x, const float* weight, const float* bias,
    float* out, int M, int N, int K) {
    // fuse matmul, bias add, and GELU tanh approx into one kernel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;

    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += x[m * K + k] * weight[n * K + k];
    acc += bias[n];
    out[idx] = 0.5 * acc * (1 + tanhf(sqrtf(2 / M_PI) * (acc + 0.044715 * acc * acc * acc)));
}
