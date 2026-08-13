// clang-format off
__device__ void accumulate_pv (const float* p_tile, const float* v_tile, float* out_acc, int tile_q,
                               int tile_k, int head_dim, int thread_id, int num_threads) {
    // clang-format on
    // cooperatively add P_tile * V_tile into out_acc
    for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads) {
        int i = idx / head_dim;
        int j = idx % head_dim;
        float acc = 0;
        for (int k = 0; k < tile_k; k++)
            acc += p_tile[i * tile_k + k] * v_tile[k * head_dim + j];
        out_acc[idx] += acc;
    }
}
