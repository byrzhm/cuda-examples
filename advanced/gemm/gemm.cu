#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

// C(MxN) = A(MxK) * B(KxN) 行优先
// 每个线程处理一个输出矩阵中的元素
__global__ void gemm_naive (const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N)
        return;

    float accum = 0.0f;
    for (int i = 0; i < K; i++) {
        accum += A[row * K + i] * B[i * N + col];
    }
    C[row * N + col] = accum;
}

void gemm_naive_bench(nvbench::state& state) {
    const int M = static_cast<int>(state.get_int64("M"));
    const int N = static_cast<int>(state.get_int64("N"));
    const int K = static_cast<int>(state.get_int64("K"));

    thrust::device_vector<float> d_A(M * K, 1.0f);
    thrust::device_vector<float> d_B(K * N, 1.0f);
    thrust::device_vector<float> d_C(M * N);

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    state.add_element_count(static_cast<std::size_t>(M) * N * K * 2);  // Multiply-add operations
    state.add_global_memory_reads<float>(static_cast<std::size_t>(M) * N * K * 2);  // Read A and B
    state.add_global_memory_writes<float>(static_cast<std::size_t>(M) * N);  // Write C

    state.exec([&d_A, &d_B, &d_C, M, N, K, grid_size, block_size](nvbench::launch& launch) {
        gemm_naive<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()),
            thrust::raw_pointer_cast(d_C.data()),
            M, N, K);
    });
}

NVBENCH_BENCH(gemm_naive_bench)
    .add_int64_axis("M", {128, 256, 512, 1024, 2048})
    .add_int64_axis("N", {128, 256, 512, 1024, 2048})
    .add_int64_axis("K", {128, 256, 512, 1024, 2048});


#define BLOCK_SIZE 32

__global__ void gemm_v1 (const float* A, const float* B, float* C, int M, int N, int K) {
    int col = blockDim.x * blockIdx.x + threadIdx.x;
    int row = blockDim.y * blockIdx.y + threadIdx.y;
    if (row >= M || col >= N)
        return;

    int bx = blockIdx.x;
    int by = blockIdx.y;
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    const int BM = BLOCK_SIZE;
    const int BN = BLOCK_SIZE;
    const int BK = BLOCK_SIZE;
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[(by * BM) * K];
    B = &B[bx * BN];
    C = &C[(by * BM) * N + bx * BN];

    float accum = 0.0f;
    for (int k = 0; k < K; k += BK) {
        // global ==> shared
        As[ty * BK + tx] = A[ty * K + tx];
        Bs[ty * BN + tx] = B[ty * N + tx];
        __syncthreads ();

        A = A + BK;
        B = B + BK * N;
        for (int i = 0; i < BK; i++) {
            accum += As[ty * BK + i] * Bs[i * BN + tx];
        }
        __syncthreads ();
    }
    C[ty * N + tx] = accum;
}

void gemm_v1_bench(nvbench::state& state) {
    const int M = static_cast<int>(state.get_int64("M"));
    const int N = static_cast<int>(state.get_int64("N"));
    const int K = static_cast<int>(state.get_int64("K"));

    // v1 requires dimensions to be multiples of BLOCK_SIZE
    if (M % BLOCK_SIZE != 0 || N % BLOCK_SIZE != 0 || K % BLOCK_SIZE != 0) {
        state.skip("Dimensions must be multiples of BLOCK_SIZE");
        return;
    }

    thrust::device_vector<float> d_A(M * K, 1.0f);
    thrust::device_vector<float> d_B(K * N, 1.0f);
    thrust::device_vector<float> d_C(M * N);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(N / BLOCK_SIZE, M / BLOCK_SIZE);

    state.add_element_count(static_cast<std::size_t>(M) * N * K * 2);
    state.add_global_memory_reads<float>(static_cast<std::size_t>(M) * N * K * 2);
    state.add_global_memory_writes<float>(static_cast<std::size_t>(M) * N);

    state.exec([&d_A, &d_B, &d_C, M, N, K, grid_size, block_size](nvbench::launch& launch) {
        gemm_v1<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()),
            thrust::raw_pointer_cast(d_C.data()),
            M, N, K);
    });
}

NVBENCH_BENCH(gemm_v1_bench)
    .add_int64_axis("M", {128, 256, 512, 1024, 2048})
    .add_int64_axis("N", {128, 256, 512, 1024, 2048})
    .add_int64_axis("K", {128, 256, 512, 1024, 2048});


template <const int BM, const int BN, const int BK, const int TM, const int TN>
__global__ void sgemm (float* A, float* B, float* C, int M, int N, int K) {
    int bx = blockIdx.x;
    int by = blockIdx.y;

    int block_row_thread = BN / TN; // block中一行的thread数量
    int block_col_thread = BM / TM; // block中一列的thread数量
    int thread_num = block_row_thread * block_col_thread; // block中thread总量

    int tx = (threadIdx.x % block_row_thread) * TN; // threadtile左上角x坐标
    int ty = (threadIdx.x / block_row_thread) * TM; // threadtile左上角y坐标

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A = &A[by * BM * K];
    B = &B[bx * BN];
    C = &C[by * BM * N + bx * BN];

    int a_tile_row = threadIdx.x / BK;
    int a_tile_col = threadIdx.x % BK;
    int a_tile_stride = thread_num / BK; // BM/(BM/(thread_num/BK)) = thread_num/BK = stride

    int b_tile_row    = threadIdx.x / BN;
    int b_tile_col    = threadIdx.x % BN;
    int b_tile_stride = thread_num / BN;

    float accum[TM][TN] = { 0.0f };
    for (int k = 0; k < K; k += BK) {
        for (int i = 0; i < BM; i += a_tile_stride) {
            As[(a_tile_row + i) * BK + a_tile_col] = A[(a_tile_row + i) * K + a_tile_col];
        }
        for (int i = 0; i < BK; i += b_tile_stride) {
            Bs[(b_tile_row + i) * BN + b_tile_col] = B[(b_tile_row + i) * N + b_tile_col];
        }
        __syncthreads ();

        A += BK;
        B += BK * N;

        for (int row = 0; row < TM; row++) {
            for (int col = 0; col < TN; col++) {
                for (int i = 0; i < BK; i++) {
                    accum[row][col] += As[(ty + row) * BK + i] * Bs[i * BN + (tx + col)];
                }
            }
        }
        __syncthreads ();
    }
    for (int row = 0; row < TM; row++) {
        for (int col = 0; col < TN; col++) {
            C[(ty + row) * N + (tx + col)] = accum[row][col];
        }
    }
}

void sgemm_bench(nvbench::state& state) {
    const int M = static_cast<int>(state.get_int64("M"));
    const int N = static_cast<int>(state.get_int64("N"));
    const int K = static_cast<int>(state.get_int64("K"));

    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    // sgemm requires dimensions to be multiples of BM, BN, BK
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        state.skip("Dimensions must satisfy M%128==0, N%128==0, K%8==0");
        return;
    }

    thrust::device_vector<float> d_A(M * K, 1.0f);
    thrust::device_vector<float> d_B(K * N, 1.0f);
    thrust::device_vector<float> d_C(M * N);

    dim3 grid_size(N / BN, M / BM);
    dim3 block_size((BM / TM) * (BN / TN));

    state.add_element_count(static_cast<std::size_t>(M) * N * K * 2);
    state.add_global_memory_reads<float>(static_cast<std::size_t>(M) * N * K * 2);
    state.add_global_memory_writes<float>(static_cast<std::size_t>(M) * N);

    state.exec([&d_A, &d_B, &d_C, M, N, K, grid_size, block_size](nvbench::launch& launch) {
        sgemm<BM, BN, BK, TM, TN><<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_B.data()),
            thrust::raw_pointer_cast(d_C.data()),
            M, N, K);
    });
}

NVBENCH_BENCH(sgemm_bench)
    .add_int64_axis("M", {128, 256, 512, 1024, 2048})
    .add_int64_axis("N", {128, 256, 512, 1024, 2048})
    .add_int64_axis("K", {128, 256, 512, 1024, 2048});
