#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <cstdio>

__global__ void sum_naive(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(output, input[idx]);
}

// assume blockDim.x is a power of 2 and N is a multiple of blockDim.x
__global__ void sum_v0(float* input, float* output) {
    int tid = threadIdx.x;
    float *x = &input[blockIdx.x * blockDim.x];

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            x[tid] += x[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = x[0];
}

__global__ void sum_v1(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[256];

    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

// Dynamic shared memory version
__global__ void sum_v2(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];

    // Load data to shared memory with boundary check
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

__global__ void sum_v3(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    extern __shared__ float sdata[];

    // Load data to shared memory with boundary check
    sdata[tid] = (idx < N) ? input[idx] : 0.0f;
    __syncthreads();

    // Reduction in shared memory
    for (int offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (tid < offset) {
            sdata[tid] += sdata[tid + offset];
        }
        __syncthreads();
    }

    if (tid == 0) atomicAdd(output, sdata[0]);
}


bool check_sum_naive(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output(1, 0.0f);

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    sum_naive<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                          thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    float result = d_output[0];
    float expected = static_cast<float>(N);

    printf("sum_naive N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}

// Warp shuffle version
// https://developer.nvidia.com/blog/using-cuda-warp-level-primitives/
__inline__ __device__ float warp_reduce(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void sum_v4(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    // extern __shared__ float sdata[];
    __shared__ float sdata[32];

    // Load data with boundary check
    float val = (idx < N) ? input[idx] : 0.0f;

    // Warp-level reduction
    val = warp_reduce(val);

    // Only first lane of each warp writes to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the warp results
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        val = warp_reduce(val);
        if (lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}

__global__ void sum_v5(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    // extern __shared__ float sdata[];
    __shared__ float sdata[32];

    // Load data with boundary check
    // float val = (idx < N) ? input[idx] : 0.0f;
    float val = 0.0f;
    if (idx < N) {
        float4 tmp = *reinterpret_cast<const float4 *>(&input[idx]);
        val += tmp.x;
        val += tmp.y;
        val += tmp.z;
        val += tmp.w;
    }

    // Warp-level reduction
    val = warp_reduce(val);

    // Only first lane of each warp writes to shared memory
    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    // First warp reduces the warp results
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        val = warp_reduce(val);
        if (lane_id == 0) {
            atomicAdd(output, val);
        }
    }
}


bool check_sum_v0(int N) {
    if (N % 256 != 0) {
        printf("sum_v0 N=%d: SKIP (N must be a multiple of 256)\n", N);
        return true; // skip this test
    }

    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    thrust::device_vector<float> d_output(grid_size, 0.0f);

    sum_v0<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_output.data()));
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        result += h_output[i];
    }
    float expected = static_cast<float>(N);

    printf("sum_v0 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}

bool check_sum_v1(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    thrust::device_vector<float> d_output(grid_size, 0.0f);

    sum_v1<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                       thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        result += h_output[i];
    }
    float expected = static_cast<float>(N);

    printf("sum_v1 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}

bool check_sum_v2(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(float);

    thrust::device_vector<float> d_output(grid_size, 0.0f);

    sum_v2<<<grid_size, block_size, shared_mem_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                                        thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = 0.0f;
    for (int i = 0; i < grid_size; ++i) {
        result += h_output[i];
    }
    float expected = static_cast<float>(N);

    printf("sum_v2 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}

bool check_sum_v3(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    size_t shared_mem_size = block_size * sizeof(float);

    thrust::device_vector<float> d_output(1, 0.0f);

    sum_v3<<<grid_size, block_size, shared_mem_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                                        thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = h_output[0];
    float expected = static_cast<float>(N);

    printf("sum_v3 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}


bool check_sum_v4(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;

    thrust::device_vector<float> d_output(1, 0.0f);

    sum_v4<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                                        thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = h_output[0];
    float expected = static_cast<float>(N);

    printf("sum_v4 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}


bool check_sum_v5(int N) {
    thrust::host_vector<float> h_input(N);
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    thrust::device_vector<float> d_input = h_input;

    int block_size = 256;
    int grid_size = (N + block_size - 1) / block_size;
    grid_size = (grid_size + 3) / 4; // since each thread processes 4 elements

    thrust::device_vector<float> d_output(1, 0.0f);

    sum_v5<<<grid_size, block_size>>>(thrust::raw_pointer_cast(d_input.data()),
                                                        thrust::raw_pointer_cast(d_output.data()), N);
    cudaDeviceSynchronize();

    thrust::host_vector<float> h_output = d_output;
    float result = h_output[0];
    float expected = static_cast<float>(N);

    printf("sum_v5 N=%d: result=%.0f, expected=%.0f, %s\n",
           N, result, expected, (result == expected) ? "PASS" : "FAIL");
    return result == expected;
}


int main() {
    printf("=== Sum correctness check ===\n\n");

    int test_sizes[] = {1024, 4096, 16384, 65536, 1000, 257, 255};

    bool all_pass = true;
    for (int N : test_sizes) {
        if (!check_sum_naive(N)) all_pass = false;
        if (!check_sum_v0(N)) all_pass = false;
        if (!check_sum_v1(N)) all_pass = false;
        if (!check_sum_v2(N)) all_pass = false;
        if (!check_sum_v3(N)) all_pass = false;
        if (!check_sum_v4(N)) all_pass = false;
        if (!check_sum_v5(N)) all_pass = false;
    }

    printf("\n=== Overall: %s ===\n", all_pass ? "ALL PASS" : "SOME FAILED");
    return all_pass ? 0 : 1;
}