#include <cfloat>
#include <cmath>


// clang-format off
__device__ float online_max(float old_max, float new_val) {
    // return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}

// clang-format off
__device__ void load_tile(const float* src, float* shared_dst,
                          int src_row_start, int src_col_start,
                          int src_rows, int src_cols,
                          int tile_rows, int tile_cols,
                          int thread_id, int num_threads) {
    // clang-format on
    // cooperatively copy the tile into shared_dst, zero-filling out-of-bounds positions.
    for (int i = thread_id; i < tile_rows; i += num_threads) {
        int row = src_row_start + i;
        for (int j = 0; j < tile_cols; j++) {
            int col = src_col_start + j;
            shared_dst[i * tile_cols + j] =
            (row < src_rows && col < src_cols) ? src[row * src_cols + col] : 0.0f;
        }
    }
}

// clang-format off
__device__ void tile_scores(const float* q_tile, const float* k_tile, float* s_tile,
                            int tile_q, int tile_k, int head_dim, float scale,
                            int thread_id, int num_threads) {
    // clang-format on
    // cooperatively fill s_tile[i, j] = scale * dot(q_tile[i, :], k_tile[j, :])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int i     = idx / tile_k;
        int j     = idx % tile_k;
        float acc = 0.0f;
        for (int k = 0; k < head_dim; k++) {
            acc += q_tile[i * head_dim + k] * k_tile[j * head_dim + k];
        }
        s_tile[idx] = acc * scale;
    }
}

__device__ void
tile_rowmax (const float* s_tile, float* row_max_out, int tile_q, int tile_k, int thread_id, int num_threads) {
    for (int r = thread_id; r < tile_q; r += num_threads) {
        float maxVal = -FLT_MAX;
        for (int c = 0; c < tile_k; c++)
            maxVal = fmaxf (maxVal, s_tile[r * tile_k + c]);
        row_max_out[r] = maxVal;
    }
}

// clang-format off
__device__ void tile_exp(float* s_tile, const float* row_max,
                         int tile_q, int tile_k,
                         int thread_id, int num_threads) {
    // clang-format on
    // for each (r, c) in the tile, set s_tile[r*tile_k+c] = expf(s_tile[r*tile_k+c] - row_max[r])
    for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
        int r = idx / tile_k;
        // int c = idx % tile_k;
        s_tile[idx] = expf (s_tile[idx] - row_max[r]);
    }
}

__device__ void
tile_rowsum (const float* p_tile, float* row_sum_out, int tile_q, int tile_k, int thread_id, int num_threads) {
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

// clang-format off
__device__ void accumulate_pv (const float* p_tile, const float* v_tile, float* out_acc, int tile_q,
                               int tile_k, int head_dim, int thread_id, int num_threads) {
    // clang-format on
    // cooperatively add P_tile * V_tile into out_acc
    for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads) {
        int i     = idx / head_dim;
        int j     = idx % head_dim;
        float acc = 0;
        for (int k = 0; k < tile_k; k++)
            acc += p_tile[i * tile_k + k] * v_tile[k * head_dim + j];
        out_acc[idx] += acc;
    }
}

__device__ float correction_factor (float old_max, float new_max) {
    // return the scalar used to rescale running statistics
    return expf (old_max - new_max);
}


__device__ float update_running_sum (float old_sum, float correction, float block_sum) {
    // combine the rescaled old sum with the new block sum
    return old_sum * correction + block_sum;
}

__device__ void rescale_output (float* out_row, int head_dim, float correction) {
    // multiply each of the head_dim entries of out_row by correction in place
    for (int i = 0; i < head_dim; i++) {
        out_row[i] *= correction;
    }
}

// clang-format off
__global__ void flash_attention_kernel(const float* q, const float* k, const float* v,
                                       float* out, int seq_len, int head_dim,
                                       int tile_q, int tile_k, float scale) {
    // tiled fused attention using shared memory and online softmax.
    const int thread_id   = threadIdx.x;
    const int num_threads = blockDim.x;
    const int block_id    = blockIdx.x;
    const int num_blocks  = gridDim.x;

    // one dynamic shared segment carved into all working tiles.
    // (multiple `extern __shared__` declarations would alias the same region.)
    extern __shared__ float smem[];
    float* q_tile   = smem;                       // tile_q * head_dim
    float* k_tile   = q_tile + tile_q * head_dim; // tile_k * head_dim
    float* v_tile   = k_tile + tile_k * head_dim; // tile_k * head_dim
    float* sp_tile  = v_tile + tile_k * head_dim; // tile_q * tile_k
    float* row_max1 = sp_tile + tile_q * tile_k;  // tile_q
    float* row_max2 = row_max1 + tile_q;          // tile_q
    float* row_sum1 = row_max2 + tile_q;          // tile_q
    float* row_sum2 = row_sum1 + tile_q;          // tile_q

    float* old_row_max = row_max1;
    float* new_row_max = row_max2;
    float* old_row_sum = row_sum1;
    float* new_row_sum = row_sum2;

    for (int q_row_start = block_id * tile_q; q_row_start < seq_len; q_row_start += num_blocks * tile_q) {
        float* out_acc = out + q_row_start * head_dim;

        // zero the output accumulator and initialise running statistics.
        for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads)
            out_acc[idx] = 0.0f;
        for (int r = thread_id; r < tile_q; r += num_threads) {
            old_row_max[r] = -FLT_MAX;
            old_row_sum[r] = 0.0f;
        }
        __syncthreads();

        // load this block's Q tile once, reused across all KV tiles.
        load_tile (q, q_tile, q_row_start, 0, seq_len, head_dim, tile_q, head_dim, thread_id, num_threads);
        __syncthreads();

        for (int kv_row_start = 0; kv_row_start < seq_len; kv_row_start += tile_k) {
            load_tile (k, k_tile, kv_row_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
            load_tile (v, v_tile, kv_row_start, 0, seq_len, head_dim, tile_k, head_dim, thread_id, num_threads);
            __syncthreads();

            // S = scale * Q @ K^T  (tile_q x tile_k)
            tile_scores (q_tile, k_tile, sp_tile, tile_q, tile_k, head_dim, scale, thread_id, num_threads);
            __syncthreads();

            // mask K positions past seq_len to -inf: load_tile zero-pads OOB K
            // rows, which would otherwise give S=0 and contaminate the softmax
            // (exp(0-max) > 0). -inf -> exp(-inf) = 0, a clean zero contribution.
            for (int idx = thread_id; idx < tile_q * tile_k; idx += num_threads) {
                int j = idx % tile_k;
                if (kv_row_start + j >= seq_len) sp_tile[idx] = -INFINITY;
            }
            __syncthreads();

            // block_max = rowmax(S)
            tile_rowmax(sp_tile, new_row_max, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // merge running max and rescale the running output to the new max.
            // corr = exp(old_max - new_max) brings old O / old l onto the new_max scale.
            for (int r = thread_id; r < tile_q; r += num_threads) {
                new_row_max[r] = online_max(old_row_max[r], new_row_max[r]);
                float corr = correction_factor(old_row_max[r], new_row_max[r]);
                rescale_output(out_acc + r * head_dim, head_dim, corr);
            }
            __syncthreads();

            // P = exp(S - new_max)  (already in the new running-max scale)
            tile_exp(sp_tile, new_row_max, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // block_sum = rowsum(P)
            tile_rowsum(sp_tile, new_row_sum, tile_q, tile_k, thread_id, num_threads);
            __syncthreads();

            // l_new = l_old * corr + block_sum
            for (int r = thread_id; r < tile_q; r += num_threads) {
                float corr = correction_factor(old_row_max[r], new_row_max[r]);
                new_row_sum[r] = update_running_sum(old_row_sum[r], corr, new_row_sum[r]);
            }
            __syncthreads();

            // O += P @ V   (P and O now share the new_max scale)
            accumulate_pv(sp_tile, v_tile, out_acc, tile_q, tile_k, head_dim, thread_id, num_threads);
            __syncthreads();

            // ping-pong: current new stats become next iteration's old stats.
            float* tmp;
            tmp         = old_row_max;
            old_row_max = new_row_max;
            new_row_max = tmp;
            tmp         = old_row_sum;
            old_row_sum = new_row_sum;
            new_row_sum = tmp;
        }

        // finalise: divide by the running sum.
        for (int idx = thread_id; idx < tile_q * head_dim; idx += num_threads) {
            int r = idx / head_dim;
            out_acc[idx] /= old_row_sum[r];
        }
        // leave out_acc fully written before the next Q tile reuses old_row_*.
        __syncthreads ();
    }
}
// clang-format on
