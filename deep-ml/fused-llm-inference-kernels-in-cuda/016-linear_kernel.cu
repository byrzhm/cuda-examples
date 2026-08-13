__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;
    
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += x[m * K + k] * weight[n * K + k];
    if (bias != nullptr)
        acc += bias[n];
    out[idx] = acc;
}
