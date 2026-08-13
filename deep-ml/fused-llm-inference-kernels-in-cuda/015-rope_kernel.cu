/*

q, k: [seq_len, n_heads, head_dim]
cos_table, sin_table: [seq_len, head_dim/2]

*/
__global__ void rope_kernel(float* q, float* k,
                            const float* cos_table, const float* sin_table,
                            int seq_len, int n_heads, int head_dim) {
    // apply RoPE rotation in-place to every even/odd pair of q and k
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = seq_len * n_heads * head_dim / 2;

    if (idx < total) {
        int t = idx / (n_heads * head_dim / 2); // token position
        int h = (idx % (n_heads * head_dim / 2)) / (head_dim / 2);
        int pair_i = idx % (head_dim / 2);
        
        int base = (t * n_heads + h) * head_dim;
        int even = base + 2 * pair_i;
        int odd = even + 1;
        float cos = cos_table[t * head_dim / 2 + pair_i];
        float sin = sin_table[t * head_dim / 2 + pair_i];

        float q_even = q[even];
        float q_odd = q[odd];
        float k_even = k[even];
        float k_odd = k[odd];
        q[even] = q_even * cos - q_odd * sin;
        q[odd] = q_even * sin + q_odd * cos;
        k[even] = k_even * cos - k_odd * sin;
        k[odd] = k_even * sin + k_odd * cos;
    }
}
