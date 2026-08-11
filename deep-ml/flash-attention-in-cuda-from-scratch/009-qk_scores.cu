__device__ float dot_product (const float* a, const float* b, int n) {
    float res = 0.0f;
    for (int i = 0; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}


// naive
__global__ void
qk_scores (const float* q, const float* k, float* scores, int seq_len, int head_dim) {
    // compute scores[i, j] = dot(q_row_i, k_row_j) / sqrt(head_dim)
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < seq_len && j < seq_len) {
        scores[i * seq_len + j] = dot_product(q + i * head_dim, k + j * head_dim, head_dim) / sqrtf(head_dim);
    }
}
