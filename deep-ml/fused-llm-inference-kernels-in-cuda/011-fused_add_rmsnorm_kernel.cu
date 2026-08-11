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

// clang-format off
__global__ void fused_add_rmsnorm_kernel(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int n,
    float eps
) {
    // fuse residual addition with RMSNorm (one block per row)
    const int tid = threadIdx.x;
    const int rowId = blockIdx.x;

    x = x + rowId * n;
    residual = residual + rowId * n;
    out = out + rowId * n;
    residual_out = residual_out + rowId * n;

    __shared__ float smem[32];
    float val;
    float sum;
    float inv_rms;

    val = 0.0f;
    for (int i = tid; i < n; i += blockDim.x) {
        float r = x[i] + residual[i];
        residual_out[i] = r;
        val += r * r;
    }

    sum = block_reduce_sum(val, smem);
    if (tid == 0) {
        smem[0] = rsqrtf(sum / n + eps);
    }
    __syncthreads();
    
    inv_rms = smem[0];

    for (int i = tid; i < n; i += blockDim.x)
        out[i] = residual_out[i] * inv_rms * weight[i];
}
// clang-format on
