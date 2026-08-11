// 可以优化
__global__ void row_sum (const float* matrix, float* out, int rows, int cols) {
    // write out[r] = sum of matrix row r
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < rows) {
        float sum = 0.0f;
        for (int col = 0; col < cols; col++) {
            sum += matrix[idx * cols + col];
        }
        out[idx] = sum;
    }
}