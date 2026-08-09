#include <iostream>
#include <cuda_runtime.h>

__global__ void printThreadInfo() {
    // 计算一维线程 ID
    int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    // 计算所属的 Warp ID
    int warpId = tid / 32;
    
    // 只打印前两个 warp 的信息，避免输出过多
    if (warpId < 2) {
        printf("Warp %d | tid=%2d | threadIdx.x=%2d | threadIdx.y=%d\n", 
               warpId, tid, threadIdx.x, threadIdx.y);
    }
}

int main() {
    std::cout << "=== Case 1: 1D Block (blockDim.x = 64) ===" << std::endl;
    dim3 block1D(64);
    printThreadInfo<<<1, block1D>>>();
    cudaDeviceSynchronize();

    std::cout << "\n=== Case 2: 2D Block (blockDim.x = 16, blockDim.y = 4) ===" << std::endl;
    dim3 block2D(16, 4);
    printThreadInfo<<<1, block2D>>>();
    cudaDeviceSynchronize();

    return 0;
}