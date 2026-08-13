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



void rmsnorm_residual_block(
    const float* x,
    const float* residual,
    const float* weight,
    float* out,
    float* residual_out,
    int rows,
    int n,
    float eps
) {
    // launch fused_add_rmsnorm_kernel for the pre-norm residual+RMSNorm block
    fused_add_rmsnorm_kernel<<<rows, 256>>>(x, residual, weight, out, residual_out, n, eps);
}

__global__ void swiglu_kernel (const float* gate, const float* up, float* out, int n) {
    // out[i] = silu(gate[i]) * up[i] for all i in [0, n)
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float gate_val = gate[idx];
        float silu_val = gate_val / (1.0f + expf (-gate_val));
        out[idx]       = silu_val * up[idx];
    }
}


__global__ void linear_kernel(const float* x, const float* weight,
                              const float* bias, float* out,
                              int M, int N, int K) {
    // compute out = x @ weight^T (+ bias if non-null)
    // x: [M*K], weight: [N*K], bias: [N] or nullptr, out: [M*N]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) return;

    int m = idx / N;
    int n = idx % N;
    
    float acc = 0.0f;
    for (int k = 0; k < K; k++)
        acc += x[m * K + k] * weight[n * K + k];
    if (bias != nullptr)
        acc += bias[n];
    out[idx] = acc;
}


void mlp_swiglu_forward(const float* x, const float* w_gate, const float* w_up,
                        const float* w_down, float* out,
                        int M, int hidden_dim, int intermediate_dim) {
    // allocate temps, run gate/up linears, swiglu, then down projection
    float* d_gate;
    float* d_up;
    float* d_act;
    cudaMalloc(&d_gate, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_up, M * intermediate_dim * sizeof(float));
    cudaMalloc(&d_act, M * intermediate_dim * sizeof(float));
    
    const int num_threads = 256;
    int num_blocks = (M * intermediate_dim + num_threads - 1) / num_threads;
    linear_kernel<<<num_blocks, num_threads>>>(
        x, w_gate, nullptr, d_gate, M, intermediate_dim, hidden_dim
    );
    linear_kernel<<<num_blocks, num_threads>>>(
        x, w_up, nullptr, d_up, M, intermediate_dim, hidden_dim
    );
    swiglu_kernel<<<num_blocks, num_threads>>>(
        d_gate, d_up, d_act, M * intermediate_dim
    );
    num_blocks = (M * hidden_dim + num_threads - 1) / num_threads;
    linear_kernel<<<num_blocks, num_threads>>>(
        d_act, w_down, nullptr, out, M, hidden_dim, intermediate_dim
    );
    cudaDeviceSynchronize();
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_act);
}


__global__ void
add_residual_kernel (const float* x, const float* residual, float* out, int n) {
    // implement elementwise residual addition out[i] = x[i] + residual[i]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
        out[idx] = x[idx] + residual[idx];
}



void run_transformer_ffn(const float* x, const float* residual,
                         const float* norm_weight, const float* w_gate,
                         const float* w_up, const float* w_down, float* out,
                         int M, int hidden_dim, int intermediate_dim,
                         float eps) {
    // residual+RMSNorm, SwiGLU MLP, then residual add into out
    float* d_residual_out;
    float* d_norm_out;
    cudaMalloc(&d_residual_out, M * hidden_dim * sizeof(float));
    cudaMalloc(&d_norm_out, M * hidden_dim * sizeof(float));

    rmsnorm_residual_block(x, residual, norm_weight, d_norm_out, d_residual_out, M, hidden_dim, eps);

    mlp_swiglu_forward(d_norm_out, w_gate, w_up, w_down, out, M, hidden_dim, intermediate_dim);

    add_residual_kernel<<<(M * hidden_dim + 255) / 256, 256>>>(
        out, d_residual_out, out, M * hidden_dim
    );

    cudaDeviceSynchronize();
    cudaFree(d_residual_out);
    cudaFree(d_norm_out);
}
