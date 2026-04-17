#include <stdio.h>

__global__ void print_arch() {
#ifdef __CUDA_ARCH__
    printf("Hello from device! __CUDA_ARCH__ = %d\n", __CUDA_ARCH__);
#else
    printf("Hello from device! __CUDA_ARCH__ not defined.\n");
#endif
}

int main() {
    print_arch<<<1, 1>>>();
    cudaDeviceSynchronize();
    return 0;
}

