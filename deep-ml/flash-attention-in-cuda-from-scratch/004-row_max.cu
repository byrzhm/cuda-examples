// 可以优化
__global__ void row_max (const float* matrix, float* out, int rows, int cols) {
    // compute the max of each row and write it to out[r].
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < rows) {
        float m = matrix[idx * cols];
        for (int col = 1; col < cols; col++) {
            m = fmaxf(m, matrix[idx * cols + col]);
        }
        out[idx] = m;
    }
}
