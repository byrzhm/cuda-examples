#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>

/*
    original size: M x N
    transposed size: N x M
*/
void transpose_cpu(const float* input, float* output, int M, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            output[j * M + i] = input[i * N + j];
        }
    }
}

__global__ void transpose_naive(const float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[col * M + row] = input[row * N + col];
    }
}

__global__ void transpose_v0(const float* input, float* output, int M, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N) {
        output[col * M + row] = __ldg(&input[row * N + col]);
    }
}

bool check_transpose_naive(const float* input, const float* expected_output, int M, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    transpose_naive<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    float*h_output = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(expected_output[i] - h_output[i]);
        if (diff > 1e-5f) {
            printf("Mismatch at index %d: CPU=%.8f, GPU=%.8f, diff=%.8f\n",
                   i, expected_output[i], h_output[i], diff);
            passed = false;
        }
    }
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return passed;
}

bool check_transpose_v0(const float* input, const float* expected_output, int M, int N) {
    float *d_input, *d_output;
    cudaMalloc(&d_input, M * N * sizeof(float));
    cudaMalloc(&d_output, M * N * sizeof(float));
    cudaMemcpy(d_input, input, M * N * sizeof(float), cudaMemcpyHostToDevice);
    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);
    transpose_v0<<<gridSize, blockSize>>>(d_input, d_output, M, N);
    float*h_output = (float*)malloc(M * N * sizeof(float));
    cudaMemcpy(h_output, d_output, M * N * sizeof(float), cudaMemcpyDeviceToHost);
    bool passed = true;
    for (int i = 0; i < M * N; i++) {
        float diff = fabsf(expected_output[i] - h_output[i]);
        if (diff > 1e-5f) {
            printf("Mismatch at index %d: CPU=%.8f, GPU=%.8f, diff=%.8f\n",
                   i, expected_output[i], h_output[i], diff);
            passed = false;
        }
    }
    free(h_output);
    cudaFree(d_input);
    cudaFree(d_output);
    return passed;
}

void randomize_matrix(float* matrix, int M, int N) {
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    for (int i = 0; i < M * N; i++) {
        matrix[i] = dist(rng);
    }
}

int main() {
    std::vector<int> test_matrix_sizes = {256, 512, 1024, 2048};
    std::vector<std::vector<int>> test_sizes;

    for (int size : test_matrix_sizes) {
        for (int size2 : test_matrix_sizes) {
            test_sizes.push_back({size, size2});
        }
    }

    bool all_passed = true;

    for (const auto& size : test_sizes) {
        int M = size[0];
        int N = size[1];
        
        float* h_input = (float*)malloc(M * N * sizeof(float));
        float* h_output_cpu = (float*)malloc(M * N * sizeof(float));

        randomize_matrix(h_input, M, N);
        transpose_cpu(h_input, h_output_cpu, M, N);

        if (!check_transpose_naive(h_input, h_output_cpu, M, N)) {
            all_passed = false;
            printf("Naive transpose failed for size %dx%d\n", M, N);
        }
        if (!check_transpose_v0(h_input, h_output_cpu, M, N)) {
            all_passed = false;
            printf("V0 transpose failed for size %dx%d\n", M, N);
        }
    }

    if (all_passed) {
        printf("All tests passed!\n");
    } else {
        printf("Some tests failed.\n");
    }

    return 0;
}