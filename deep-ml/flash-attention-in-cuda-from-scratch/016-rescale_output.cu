__device__ void rescale_output(float* out_row, int head_dim, float correction) {
    // multiply each of the head_dim entries of out_row by correction in place
    for (int i = 0; i < head_dim; i++) {
        out_row[i] *= correction;
    }
}
