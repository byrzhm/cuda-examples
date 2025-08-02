#include <stdio.h>
#include <cuda_runtime.h>

__global__ void reinterpret_kernel(const long long *input, double *output) {
    int idx = threadIdx.x;
    output[idx] = __longlong_as_double(input[idx]);
}

int main() {
    const int N = 4;
    long long h_input[N] = {
        0x3ff0000000000000,  // Bit pattern of 1.0 in IEEE-754 double
        0x4000000000000000,  // 2.0
        0x4008000000000000,  // 3.0
        0x4010000000000000   // 4.0
    };
    double h_output[N];

    long long *d_input;
    double *d_output;
    cudaMalloc(&d_input, N * sizeof(long long));
    cudaMalloc(&d_output, N * sizeof(double));

    cudaMemcpy(d_input, h_input, N * sizeof(long long), cudaMemcpyHostToDevice);

    reinterpret_kernel<<<1, N>>>(d_input, d_output);

    cudaMemcpy(h_output, d_output, N * sizeof(double), cudaMemcpyDeviceToHost);

    printf("Reinterpreted results:\n");
    for (int i = 0; i < N; ++i) {
        printf("0x%llx -> %f\n", h_input[i], h_output[i]);
    }

    cudaFree(d_input);
    cudaFree(d_output);
    return 0;
}

