#include <cfloat>

__device__ float warp_reduce_max(float val) {
    int mask = __activemask();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, off));
    return val;
}

__device__ float safe_warp_reduce_max(float val) {
    int mask = __activemask();
    int active = __popc(mask);          // 活跃线程数
    int max_off = 1;
    while (max_off < active) max_off <<= 1;
    max_off >>= 1;                      // 小于等于 active 的最大 2 的幂
    for (int off = max_off; off > 0; off >>= 1)
        val = fmaxf(val, __shfl_down_sync(mask, val, off));
    return val;
}

__device__ void
tile_rowmax (const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    // write row_max_out[r] = max over c of s_tile[r, c]
    const int numWarps = (num_threads + warpSize - 1) / warpSize;
    const int warpId   = thread_id / warpSize;
    const int laneId   = thread_id % warpSize;

    // one warp per row
    const int iterations = (tile_k + warpSize - 1) / warpSize;
    for (int r = warpId; r < tile_q; r += numWarps) {
        float maxVal = -FLT_MAX;
        for (int it = 0; it < iterations; it++) {
            int c     = it * warpSize + laneId;
            float val = (c < tile_k) ? s_tile[r * tile_k + c] : -FLT_MAX;
            float warpMaxVal = safe_warp_reduce_max(val);
            if (laneId == 0) maxVal = fmaxf(maxVal, warpMaxVal);
        }
        if (laneId == 0)
            row_max_out[r] = maxVal;
    }
}

/*
__device__ void
tile_rowmax (const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float maxVal = -FLT_MAX;
        for (int c = 0; c < tile_k; c++)
            maxVal = fmaxf(maxVal, s_tile[r * tile_k + c]);
        row_max_out[r] = maxVal;
    }
}
*/