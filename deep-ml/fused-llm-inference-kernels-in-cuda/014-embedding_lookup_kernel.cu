__global__ void embedding_lookup_kernel (const int* token_ids, const float* weight, float* out,
                                         int seq_len, int vocab_size, int embed_dim) {
    // gather embedding vectors for each token id into out
    // token_ids: [seq_len,]
    // weight: [vocab_size, embed_dim]
    // out: [seq_len, embed_dim]

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < seq_len * embed_dim) {
        int token_pos = idx / embed_dim;
        int dim = idx % embed_dim;
        int token_id = token_ids[token_pos];

        // out[i * D + d] = weight[token_ids[i] * D + d]
        out[idx] = weight[token_id * embed_dim + dim];
    }
}
