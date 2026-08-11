__device__ float warp_reduce_sum (float val) {
    // implement warp-level sum reduction using shuffle intrinsics
    int mask = __activemask ();
    for (int off = warpSize / 2; off > 0; off >>= 1)
        val += __shfl_down_sync (mask, val, off);
    val = __shfl_sync (mask, val, 0);
    return val;
}

__device__ float block_reduce_sum (float val, float* shared) {
    // block-level sum via warp_reduce_sum + shared memory; result valid on thread 0
    int tid      = threadIdx.x;
    int laneId   = tid % warpSize;
    int warpId   = tid / warpSize;
    int numWarps = (blockDim.x + warpSize - 1) / warpSize;

    float sum = warp_reduce_sum (val);
    if (laneId == 0) {
        shared[warpId] = sum;
    }
    __syncthreads ();

    if (warpId == 0) {
        sum = (laneId < numWarps) ? shared[laneId] : 0.0f;
        sum = warp_reduce_sum (sum);
    }
    return sum;
}

__global__ void
rmsnorm_kernel (const float* x, const float* weight, float* out, int n, float eps) {
    // Apply RMSNorm per row (one block per row)
    int tid    = threadIdx.x;
    const int rowId = blockIdx.x;
    const float *x_row = x + rowId * n;
    float *out_row = out + rowId * n;

    __shared__ float smem[32];


    float sum_sq = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        sum_sq += x_row[i] * x_row[i];
    }

    float sum = block_reduce_sum (sum_sq, smem);
    if (tid == 0) {
        smem[0] = rsqrtf (sum / n + eps);
    }
    __syncthreads ();

    float inv_rms = smem[0];

    for (int i = tid; i < n; i += blockDim.x) {
        out_row[i] = x_row[i] * weight[i] * inv_rms;
    }
}
