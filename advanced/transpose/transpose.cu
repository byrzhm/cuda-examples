#include <numeric>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>


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

void transpose_cpu_bench(nvbench::state& state) {
    const std::size_t M = static_cast<std::size_t>(state.get_int64("M"));
    const std::size_t N = static_cast<std::size_t>(state.get_int64("N"));
    thrust::host_vector<float> h_input(M * N);
    thrust::host_vector<float> h_output(M * N);

    std::iota(h_input.begin(), h_input.end(), 0.0f);

    state.exec([&](nvbench::launch& launch) {
        transpose_cpu(thrust::raw_pointer_cast(h_input.data()), thrust::raw_pointer_cast(h_output.data()), M, N);
    });
}

NVBENCH_BENCH(transpose_cpu_bench)
    .add_int64_axis("M", {1 << 10, 1 << 12, 1 << 14})
    .add_int64_axis("N", {1 << 10, 1 << 12, 1 << 14});

void transpose_naive_bench(nvbench::state& state) {
    const std::size_t M = static_cast<std::size_t>(state.get_int64("M"));
    const std::size_t N = static_cast<std::size_t>(state.get_int64("N"));
    thrust::host_vector<float> h_input(M * N);
    std::iota(h_input.begin(), h_input.end(), 0.0f);

    thrust::device_vector<float> d_input(M * N);
    thrust::device_vector<float> d_output(M * N);
    cudaMemcpy(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(h_input.data()), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    state.exec([&](nvbench::launch& launch) {
        transpose_naive<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), M, N);
    });
}

NVBENCH_BENCH(transpose_naive_bench)
    .add_int64_axis("M", {1 << 10, 1 << 12, 1 << 14})
    .add_int64_axis("N", {1 << 10, 1 << 12, 1 << 14});

void transpose_v0_bench(nvbench::state& state) {
    const std::size_t M = static_cast<std::size_t>(state.get_int64("M"));
    const std::size_t N = static_cast<std::size_t>(state.get_int64("N"));
    thrust::host_vector<float> h_input(M * N);
    std::iota(h_input.begin(), h_input.end(), 0.0f);
    thrust::device_vector<float> d_input(M * N);
    thrust::device_vector<float> d_output(M * N);
    cudaMemcpy(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(h_input.data()), M * N * sizeof(float), cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x, (M + blockSize.y - 1) / blockSize.y);

    state.exec([&](nvbench::launch& launch) {
        transpose_v0<<<gridSize, blockSize>>>(thrust::raw_pointer_cast(d_input.data()), thrust::raw_pointer_cast(d_output.data()), M, N);
    });
}

NVBENCH_BENCH(transpose_v0_bench)
    .add_int64_axis("M", {1 << 10, 1 << 12, 1 << 14})
    .add_int64_axis("N", {1 << 10, 1 << 12, 1 << 14});