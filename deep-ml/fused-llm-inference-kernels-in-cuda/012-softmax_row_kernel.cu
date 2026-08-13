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

__device__ float warp_reduce_sum (float val) {
    // implement warp-level sum reduction using shuffle intrinsics
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    val = __shfl_sync (mask, val, 0);
    return val;
}


__device__ float block_reduce_sum(float val, float* shared) {
    // block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int tid = threadIdx.x;
    int laneId = tid % warpSize;
    int warpId = tid / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    float sum = warp_reduce_sum(val);
    if (laneId == 0) {
        shared[warpId] = sum;
    }
    __syncthreads();

    if (warpId == 0) {
        sum = (laneId < numWarps) ? shared[laneId] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    return sum;
}


__global__ void softmax_row_kernel (const float* x, float* out, int rows, int cols) {
    // implement numerically stable row-wise softmax (one block per row)
    const int rowId = blockIdx.x;
    const int tid = threadIdx.x;
    const int numThreads = blockDim.x;

    extern __shared__ float smem[];
    
    const float *x_row = x + rowId * cols;
    float *out_row = out + rowId * cols;
    float val;
    float row_max;
    float row_sum;

    val = -FLT_MAX;
    for (int i = tid; i < cols; i += numThreads)
        val = fmaxf(val, x_row[i]);
    row_max = block_reduce_max(val, smem);
    if (tid == 0)
        smem[0] = row_max; 
    __syncthreads();
    row_max = smem[0];
    
    val = 0.0f;
    for (int i = tid; i < cols; i += numThreads)
        val += expf(x_row[i] - row_max);
    row_sum = block_reduce_sum(val, smem);
    if (tid == 0)
        smem[0] = row_sum;
    __syncthreads();
    row_sum = smem[0];

    for (int i = tid; i < cols; i += numThreads)
        out_row[i] = expf(x_row[i] - row_max) / row_sum;
}
