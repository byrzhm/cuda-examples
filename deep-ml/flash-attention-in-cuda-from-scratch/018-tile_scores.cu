// clang-format off
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;
        float acc = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            acc += q_tile[i * head_dim + k] * k_tile[j * head_dim + k];
        }
        s_tile[idx] = acc * scale;
    }
}
