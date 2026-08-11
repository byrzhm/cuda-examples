__global__ void swiglu_kernel (const float* gate, const float* up, float* out, int n) {
    // out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate_val = gate[idx];
        float silu_val = gate_val / (1.0f + expf (-gate_val));
        out[idx]       = silu_val * up[idx];
    }
}
