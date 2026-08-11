#include <cfloat>

__device__ float warp_reduce_max (float val) {
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val = fmaxf (val, __shfl_down_sync (mask, val, off));
    return val; // only laneId == 0
}

__device__ float warp_reduce_sum (float val) {
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    return val; // only laneId == 0
}

__device__ float block_reduce_max (float val, float* shared) {
    const int tid     = threadIdx.x;
    const int laneId  = tid % warpSize;
    const int warpId  = tid / warpSize;
    const int warpNum = blockDim.x / warpSize;

    val = warp_reduce_max (val);
    if (laneId == 0)
        shared[warpId] = val;
    __syncthreads ();

    if (warpId == 0) {
        val = (laneId < warpNum) ? shared[laneId] : -FLT_MAX;
        val = warp_reduce_max (val);
    }
    return val; // only tid == 0
}

__device__ float block_reduce_sum (float val, float* shared) {
    const int tid     = threadIdx.x;
    const int laneId  = tid % warpSize;
    const int warpId  = tid / warpSize;
    const int warpNum = blockDim.x / warpSize;

    val = warp_reduce_sum (val);
    if (laneId == 0)
        shared[warpId] = val;
    __syncthreads ();

    if (warpId == 0) {
        val = (laneId < warpNum) ? shared[laneId] : 0.0f;
        val = warp_reduce_sum (val);
    }
    return val; // only tid == 0
}


__global__ void softmax_rows (float* matrix, int rows, int cols) {
    // implement numerically stable row-wise softmax in place
    const int tid    = threadIdx.x;
    const int rowIdx = blockIdx.x;
    float* row       = matrix + rowIdx * cols;

    __shared__ float smem[32];
    float maxVal;
    float val;
    float sum;
    float inv_sum;

    maxVal = -FLT_MAX;
    for (int i = tid; i < cols; i += blockDim.x)
        maxVal = fmaxf(maxVal, row[i]);
    maxVal = block_reduce_max(maxVal, smem);
    if (tid == 0)
        smem[0] = maxVal;
    __syncthreads();

    maxVal = smem[0];

    val = 0.0f;
    for (int i = tid; i < cols; i += blockDim.x)
        val += expf(row[i] - maxVal);
    sum = block_reduce_sum(val, smem);
    if (tid == 0)
        smem[0] = 1 / sum;
    __syncthreads();

    inv_sum = smem[0];

    for (int i = tid; i < cols; i += blockDim.x)
        row[i] = expf(row[i] - maxVal) * inv_sum;
}
