#include <cfloat>

__device__ float warp_reduce_max (float val) {
    // implement warp-level max reduction using shuffle intrinsics
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val = fmaxf (val, __shfl_down_sync (mask, val, off));
    val = __shfl_sync (mask, val, 0);
    return val;
}

__device__ float block_reduce_max (float val, float* shared) {
    // block-wide max via warp_reduce_max + shared memory
    int tid      = threadIdx.x;
    int laneId   = tid % warpSize;
    int warpId   = tid / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    float maxVal = warp_reduce_max (val);
    if (laneId == 0) {
        shared[warpId] = maxVal;
    }
    __syncthreads ();

    if (warpId == 0) {
        maxVal = (laneId < numWarps) ? shared[laneId] : -FLT_MAX;
        maxVal = warp_reduce_max (maxVal);
    }
    return maxVal;
}
