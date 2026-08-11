__global__ void
pv_matmul (const float* p, const float* v, float* out, int seq_len, int head_dim) {
    // compute out[i, d] = sum_j p[i, j] * v[j, d]
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int d = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < seq_len && d < head_dim) {
        float acc = 0.0f;
        for (int j = 0; j < seq_len; j++) {
            acc += p[i * seq_len + j] * v[j * head_dim + d];
        }
        out[i * head_dim + d] = acc;
    }
}
