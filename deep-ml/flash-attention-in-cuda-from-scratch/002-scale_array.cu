__global__ void scale_array(float* a, float scalar, int n) {
    // multiply each element of a by scalar in place
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        a[idx] = a[idx] * scalar;
    }
}
