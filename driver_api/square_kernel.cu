extern "C" __global__ void square(float *data, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        data[idx] = data[idx] * data[idx];
}
