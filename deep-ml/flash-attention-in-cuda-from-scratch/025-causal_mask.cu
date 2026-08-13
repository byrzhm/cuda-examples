// clang-format off
__device__ void causal_mask(float* s_tile, int q_row_start, int k_col_start,
                            int tile_q, int tile_k, int thread_id, int num_threads) {
    // clang-format on
    // write -INFINITY into entries where the global key index exceeds the global query index.
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int i = idx / tile_k;
        int j = idx % tile_k;
        int global_i = q_row_start + i;
        int global_j = k_col_start + j;
        if (global_i < global_j) s_tile[idx] = -INFINITY;
    }
}
