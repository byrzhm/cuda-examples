__global__ void swiglu_kernel (const float* gate, const float* up, float* out, int n) {
    // out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate_val = gate[idx];
        float silu_val = gate_val / (1.0f + expf (-gate_val));
        out[idx]       = silu_val * up[idx];
    }
}


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


void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // allocate temps, run gate/up linears, swiglu, then down projection
    float* d_gate;
    float* d_up;
    float* d_act;
    cudaMalloc(&d_gate, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_act, M * intermediate_dim * sizeof(float));
    
    const int num_threads = 256;
    int num_blocks = (M * intermediate_dim + num_threads - 1) / num_threads;
    linear_kernel<<<num_blocks, num_threads>>>(
        x, w_gate, nullptr, d_gate, M, intermediate_dim, hidden_dim
    );
    linear_kernel<<<num_blocks, num_threads>>>(
        x, w_up, nullptr, d_up, M, intermediate_dim, hidden_dim
    );
    swiglu_kernel<<<num_blocks, num_threads>>>(
        d_gate, d_up, d_act, M * intermediate_dim
    );
    num_blocks = (M * hidden_dim + num_threads - 1) / num_threads;
    linear_kernel<<<num_blocks, num_threads>>>(
        d_act, w_down, nullptr, out, M, hidden_dim, intermediate_dim
    );
    cudaDeviceSynchronize();
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_act);
}
