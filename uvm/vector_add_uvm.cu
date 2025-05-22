#include <iostream>
#include <cuda_runtime.h>

__global__ void addVectors(int *a, int *b, int *c, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N)
        c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;
    int *a, *b, *c;

    // Allocate unified memory (accessible by CPU and GPU)
    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));

    // Initialize memory on host (CPU)
    for (int i = 0; i < N; i++) {
        a[i] = i;
        b[i] = 2 * i;
    }

    // Launch kernel on GPU
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    addVectors<<<numBlocks, blockSize>>>(a, b, c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Check the result on host
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (c[i] != a[i] + b[i]) {
            success = false;
            std::cerr << "Mismatch at index " << i << ": "
                      << c[i] << " != " << a[i] + b[i] << std::endl;
            break;
        }
    }

    if (success)
        std::cout << "Vector addition successful!\n";

    // Free unified memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
