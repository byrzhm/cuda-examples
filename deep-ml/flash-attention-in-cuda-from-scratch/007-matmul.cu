// 可以优化
__global__ void matmul (const float* a, const float* b, float* c, int m, int k, int n) {
    // compute C = A * B for row-major matrices
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    int col = blockDim.x * blockIdx.x + threadIdx.x;

    if (row < m && col < n) {
        float acc = 0.0f;
        for (int i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i * n + col];
        }
        c[row * n + col] = acc;
    }
}
