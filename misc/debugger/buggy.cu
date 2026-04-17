#include <iostream>

__global__ void buggy_kernel (int* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // Bug: out-of-bounds access for demonstration
        data[idx + 1] = idx;
    }
}

int main () {
    const int N = 16;
    int* d_data;

    cudaMalloc (&d_data, N * sizeof (int));
    buggy_kernel<<<1, N>>> (d_data, N);
    cudaDeviceSynchronize ();

    int h_data[N];
    cudaMemcpy (h_data, d_data, N * sizeof (int), cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        std::cout << h_data[i] << " ";
    }
    std::cout << std::endl;

    cudaFree (d_data);
    return 0;
}
