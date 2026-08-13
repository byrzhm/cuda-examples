// clang-format off
extern __global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale);
// clang-format on

// clang-format off
void flash_attention_launcher(const float* d_q, const float* d_k, const float* d_v,
                              float* d_out, int seq_len, int head_dim,
                              int tile_q, int tile_k) {
    // clang-format on
    // configure grid/block/shared memory and launch flash_attention_kernel
    dim3 blockDim(256);
    dim3 gridDim((seq_len + tile_q - 1) / tile_q);

    float scale = 1 / sqrtf(head_dim);
    int sharedMemSize = (tile_q * tile_k + tile_q * head_dim + 2 * tile_k * head_dim + 4 * tile_q) * sizeof(float);

    flash_attention_kernel<<<gridDim, blockDim, sharedMemSize>>>(d_q, d_k, d_v, d_out, seq_len, head_dim, tile_q, tile_k, scale);
}
