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
layernorm_kernel (const float* x, const float* weight, const float* bias, float* out, int n, float eps) {
    // per-row LayerNorm using block_reduce_sum for mean and variance
    const int tid   = threadIdx.x;
    const int rowId = blockIdx.x;

    const float* x_row = x + rowId * n;
    float* out_row     = out + rowId * n;

    float val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x)
        val += x_row[i];

    __shared__ float smem[32];

    float sum = block_reduce_sum (val, smem);
    if (tid == 0)
        smem[0] = sum / n;
    __syncthreads ();

    float mean = smem[0];

    val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float t = x_row[i] - mean;
        val += t * t;
    }
    sum = block_reduce_sum (val, smem);
    if (tid == 0) {
        smem[0] = rsqrtf (sum / n + eps);
    }
    __syncthreads ();

    float inv_std = smem[0];

    for (int i = tid; i < n; i += blockDim.x)
        out_row[i] = (x_row[i] - mean) * inv_std * weight[i] + bias[i];
}
