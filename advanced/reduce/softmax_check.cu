#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

void softmax_cpu(const float* input, float* output, int N) {
    float max_val = input[0];
    for (int i = 1; i < N; i++) {
        max_val = std::max(max_val, input[i]);
    }

    float sum_exp = 0.0f;
    for (int i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum_exp += output[i];
    }

    for (int i = 0; i < N; i++) {
        output[i] /= sum_exp;
    }
}

__inline__ __device__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val = fmax(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

__device__ static float atomicMax(float* address, float val) {
    int* address_as_i = (int*)address;
    int old = *address_as_i, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_i, assumed,
            __float_as_int(fmax(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void reduce_max(const float* input, float* max_val, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    __shared__ float sdata[32];

    float val = (idx < N) ? input[idx] : (-FLT_MAX);

    val = warp_reduce_max(val);

    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? sdata[lane_id] : (-FLT_MAX);
        val = warp_reduce_max(val);
        if (lane_id == 0) {
            atomicMax(max_val, val);
        }
    }
}

__inline__ __device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__global__ void softmax_exp_sum(
    const float* input, const float* max_val, float* sum_val, int N
) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    __shared__ float sdata[32];

    float val = (idx < N) ? expf(input[idx] - *max_val) : 0.0f;

    val = warp_reduce_sum(val);

    if (lane_id == 0) {
        sdata[warp_id] = val;
    }
    __syncthreads();

    if (warp_id == 0) {
        val = (lane_id < num_warps) ? sdata[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
        if (lane_id == 0) {
            atomicAdd(sum_val, val);
        }
    }
}

__global__ void softmax_normalize(
    const float* input, const float* max_val,
    const float* sum_val, float* output, int N
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) output[idx] = expf(input[idx] - *max_val) / (*sum_val);
}

void softmax_gpu(const float* d_input, float* d_output, float* d_max, float* d_sum, int N) {
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    cudaMemset(d_max, 0, sizeof(float));
    cudaMemset(d_sum, 0, sizeof(float));

    reduce_max<<<numBlocks, blockSize>>>(d_input, d_max, N);
    softmax_exp_sum<<<numBlocks, blockSize>>>(d_input, d_max, d_sum, N);
    softmax_normalize<<<numBlocks, blockSize>>>(d_input, d_max, d_sum, d_output, N);
}

bool verify_result(const float* ref, const float* out, int N, float epsilon = 1e-5f) {
    for (int i = 0; i < N; i++) {
        float diff = fabsf(ref[i] - out[i]);
        if (diff > epsilon) {
            printf("Mismatch at index %d: CPU=%.8f, GPU=%.8f, diff=%.8f\n",
                   i, ref[i], out[i], diff);
            return false;
        }
    }
    return true;
}

int main() {
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    float* h_input = (float*)malloc(bytes);
    float* h_output_gpu = (float*)malloc(bytes);
    float* h_output_cpu = (float*)malloc(bytes);

    srand(42);
    for (int i = 0; i < N; i++) {
        h_input[i] = (float)rand() / RAND_MAX * 10.0f - 5.0f;
    }

    float *d_input, *d_output, *d_max, *d_sum;
    cudaMalloc(&d_input, bytes);
    cudaMalloc(&d_output, bytes);
    cudaMalloc(&d_max, sizeof(float));
    cudaMalloc(&d_sum, sizeof(float));

    cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice);

    softmax_gpu(d_input, d_output, d_max, d_sum, N);

    cudaMemcpy(h_output_gpu, d_output, bytes, cudaMemcpyDeviceToHost);

    softmax_cpu(h_input, h_output_cpu, N);

    bool passed = verify_result(h_output_cpu, h_output_gpu, N);

    if (passed) {
        printf("Test PASSED: GPU softmax matches CPU reference\n");
    } else {
        printf("Test FAILED: GPU softmax does not match CPU reference\n");
    }

    float sum_cpu = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_cpu += h_output_cpu[i];
    }
    printf("CPU softmax sum: %.6f (should be ~1.0)\n", sum_cpu);

    float sum_gpu = 0.0f;
    for (int i = 0; i < N; i++) {
        sum_gpu += h_output_gpu[i];
    }
    printf("GPU softmax sum: %.6f (should be ~1.0)\n", sum_gpu);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaFree(d_max);
    cudaFree(d_sum);
    free(h_input);
    free(h_output_gpu);
    free(h_output_cpu);

    return passed ? 0 : 1;
}