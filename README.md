# CUDA Examples

## Directory Structure

```
.
├── misc/          # Miscellaneous CUDA code snippets and examples
├── advanced/      # Advanced CUDA optimization techniques
└── multi_gpu/     # Multi-GPU programming examples
```

## misc/

Miscellaneous CUDA examples:

- **debugger** - CUDA debugger examples
- **driver_api** - CUDA Driver API examples
- **nvcc_macro** - NVCC compiler macros
- **pytorch** - PyTorch CUDA extensions
- **type_cast** - CUDA type casting examples
- **uvm** - Unified Virtual Memory examples

## CUDA Code Snippets

### Kernel Definition and Execution

A kernel is the function that runs on the GPU. You define it with `__global__` and launch it using the `<<<...>>>` execution configuration.

- Kernel Definition:

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int numElements) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < numElements) {
        C[i] = A[i] + B[i];
    }
}
```

- Kernel Launch:

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (numElements + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, numElements);
```


### Device Memory Management

Data must be moved from the CPU (Host) to the GPU (Device) before processing.

```cpp
float *d_A;
size_t size = numElements * sizeof(float);
cudaMalloc((void**)&d_A, size); // Allocate device memory
cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice); // Copy to device
// ... run kernel ...
cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost); // Copy back to host
cudaFree(d_A); // Free memory
```

### Error Checking Macro

Standard practice involves wrapping CUDA API calls in a macro to catch runtime failures.

```cpp
#define CHECK_CUDA(call) { \
    const cudaError_t error = call; \
    if (error != cudaSuccess) { \
        printf("Error: %s:%d, ", __FILE__, __LINE__); \
        printf("code: %d, reason: %s\n", error, cudaGetErrorString(error)); \
        exit(1); \
    } \
}
// Usage: CHECK_CUDA(cudaMalloc(&d_ptr, size));
```

### Unified Memory (Managed Memory)

```cpp
float *data;
cudaMallocManaged(&data, size); // Allocate managed memory
// 'data' can now be accessed by both host and device code
cudaFree(data);
```