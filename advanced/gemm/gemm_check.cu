#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>
#include <cmath>

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

// CPU reference implementation for verification
void gemm_cpu(const float* A, const float* B, float* C, int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

// Check if two matrices are approximately equal
bool matrix_equal(const float* A, const float* B, int size, float epsilon = 1e-3) {
    for (int i = 0; i < size; i++) {
        if (std::fabs(A[i] - B[i]) > epsilon) {
            return false;
        }
    }
    return true;
}

bool check_gemm_naive(int M, int N, int K) {
    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C_ref(M * N);

    // Initialize with simple values
    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 10) / 10.0f;

    // CPU reference
    gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    // GPU computation
    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M * N);

    dim3 block_size(16, 16);
    dim3 grid_size((N + block_size.x - 1) / block_size.x,
                   (M + block_size.y - 1) / block_size.y);

    gemm_naive<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        M, N, K);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_C = d_C;

    bool passed = matrix_equal(h_C.data(), h_C_ref.data(), M * N);
    printf("gemm_naive M=%d, N=%d, K=%d: %s\n", M, N, K, passed ? "PASS" : "FAIL");
    return passed;
}

bool check_gemm_v1(int M, int N, int K) {
    // v1 requires dimensions to be multiples of BLOCK_SIZE
    if (M % BLOCK_SIZE != 0 || N % BLOCK_SIZE != 0 || K % BLOCK_SIZE != 0) {
        printf("gemm_v1 M=%d, N=%d, K=%d: SKIP (requires multiples of %d)\n", M, N, K, BLOCK_SIZE);
        return true;
    }

    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C_ref(M * N);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 10) / 10.0f;

    gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M * N);

    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size(N / BLOCK_SIZE, M / BLOCK_SIZE);

    gemm_v1<<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        M, N, K);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_C = d_C;

    bool passed = matrix_equal(h_C.data(), h_C_ref.data(), M * N);
    printf("gemm_v1 M=%d, N=%d, K=%d: %s\n", M, N, K, passed ? "PASS" : "FAIL");
    return passed;
}

bool check_sgemm(int M, int N, int K) {
    const int BM = 128, BN = 128, BK = 8, TM = 8, TN = 8;

    // sgemm requires dimensions to be multiples of BM, BN, BK
    if (M % BM != 0 || N % BN != 0 || K % BK != 0) {
        printf("sgemm M=%d, N=%d, K=%d: SKIP (requires M%%128==0, N%%128==0, K%%8==0)\n", M, N, K);
        return true;
    }

    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(K * N);
    thrust::host_vector<float> h_C_ref(M * N);

    for (int i = 0; i < M * K; i++) h_A[i] = static_cast<float>(i % 10) / 10.0f;
    for (int i = 0; i < K * N; i++) h_B[i] = static_cast<float>(i % 10) / 10.0f;

    gemm_cpu(h_A.data(), h_B.data(), h_C_ref.data(), M, N, K);

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C(M * N);

    dim3 grid_size(N / BN, M / BM);
    dim3 block_size((BM / TM) * (BN / TN));

    sgemm<BM, BN, BK, TM, TN><<<grid_size, block_size>>>(
        thrust::raw_pointer_cast(d_A.data()),
        thrust::raw_pointer_cast(d_B.data()),
        thrust::raw_pointer_cast(d_C.data()),
        M, N, K);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_C = d_C;

    bool passed = matrix_equal(h_C.data(), h_C_ref.data(), M * N);
    printf("sgemm M=%d, N=%d, K=%d: %s\n", M, N, K, passed ? "PASS" : "FAIL");
    return passed;
}

int main() {
    printf("=== GEMM correctness check ===\n\n");

    // Test configurations: M, N, K
    struct TestCase {
        int M, N, K;
    };

    TestCase test_cases[] = {
        {64, 64, 64},
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {100, 100, 100},  // Non-power-of-2
        {63, 127, 65},    // Odd sizes
    };

    bool all_pass = true;
    for (const auto& tc : test_cases) {
        if (!check_gemm_naive(tc.M, tc.N, tc.K)) all_pass = false;
        if (!check_gemm_v1(tc.M, tc.N, tc.K)) all_pass = false;
        if (!check_sgemm(tc.M, tc.N, tc.K)) all_pass = false;
        printf("\n");
    }

    printf("=== Overall: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}
