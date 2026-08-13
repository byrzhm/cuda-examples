__device__ float warp_reduce_sum (float val) {
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    return val; // only laneId == 0
}

__device__ float safe_warp_reduce_sum (float val) {
    int mask    = __activemask ();
    int active  = __popc (mask); // 活跃线程数
    int max_off = 1;
    while (max_off < active)
        max_off <<= 1;
    max_off >>= 1; // 小于等于 active 的最大 2 的幂
    for (int off = max_off; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    return val;
}

// clang-format off
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill row_sum_out[r] with the sum of p_tile row r
    const int numWarps = (num_threads + warpSize - 1) / warpSize;
    const int warpId   = thread_id / warpSize;
    const int laneId   = thread_id % warpSize;

    // one warp per row
    const int iterations = (tile_k + warpSize - 1) / warpSize;
    for (int r = warpId; r < tile_q; r += numWarps) {
        float row_sum = 0.0f;
        for (int it = 0; it < iterations; it++) {
            int c          = it * warpSize + laneId;
            float val      = (c < tile_k) ? p_tile[r * tile_k + c] : 0.0f;
            float warp_sum = safe_warp_reduce_sum (val);
            if (laneId == 0)
                row_sum += warp_sum;
        }
        if (laneId == 0)
            row_sum_out[r] = row_sum;
    }
}

/*

// clang-format off
__device__ void tile_rowsum(const float* p_tile, float* row_sum_out,
                            int tile_q, int tile_k,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill row_sum_out[r] with the sum of p_tile row r

    // one thread per row
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float row_sum = 0.0f;
        for (int c = 0; c < tile_k; c++) {
            row_sum += p_tile[r * tile_k + c];
        }
        row_sum_out[r] = row_sum;
    }
}

*/