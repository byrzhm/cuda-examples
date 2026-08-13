// clang-format off
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // clang-format on
    // for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int r = idx / tile_k;
        // int c = idx % tile_k;
        s_tile[idx] = expf(s_tile[idx] - row_max[r]);
    }
}
