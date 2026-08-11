// 可以优化
__global__ void transpose(const float* in, float* out, int rows, int cols) {
    // write out[c*rows + r] = in[r*cols + c]
    int r = blockDim.y * blockIdx.y + threadIdx.y;
    int c = blockDim.x * blockIdx.x + threadIdx.x;

    if (r < rows && c < cols) {
        out[c * rows + r] = in[r * cols + c];
    }
}
