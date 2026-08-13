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
            shared_dst[i * tile_cols + j] = (col < src_cols) ? src[row * src_cols + col] : 0.0f;
        }
    }
}
