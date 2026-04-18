#include <cfloat>
#include <cmath>
#include <cstdio>
#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

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

void softmax_cpu_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    thrust::host_vector<float> input(n);
    thrust::host_vector<float> output(n);

    for (std::size_t i = 0; i < n; ++i) {
        input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    state.add_element_count(n);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n](nvbench::launch&) {
        softmax_cpu(thrust::raw_pointer_cast(input.data()),
                    thrust::raw_pointer_cast(output.data()),
                    static_cast<int>(n));
    });
}

NVBENCH_BENCH(softmax_cpu_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});

void softmax_gpu_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    thrust::host_vector<float> h_input(n);
    for (std::size_t i = 0; i < n; ++i) {
        h_input[i] = static_cast<float>(rand()) / RAND_MAX * 10.0f - 5.0f;
    }

    thrust::device_vector<float> d_input = h_input;
    thrust::device_vector<float> d_output(n);
    thrust::device_vector<float> d_max(1);
    thrust::device_vector<float> d_sum(1);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(n);

    state.exec(nvbench::exec_tag::sync,
        [&d_input, &d_output, &d_max, &d_sum, n, block_size, grid_size](nvbench::launch& launch) {
            cudaMemset(thrust::raw_pointer_cast(d_max.data()), 0, sizeof(float));
            cudaMemset(thrust::raw_pointer_cast(d_sum.data()), 0, sizeof(float));

            reduce_max<<<grid_size, block_size, 0, launch.get_stream()>>>(
                thrust::raw_pointer_cast(d_input.data()),
                thrust::raw_pointer_cast(d_max.data()),
                static_cast<int>(n));

            softmax_exp_sum<<<grid_size, block_size, 0, launch.get_stream()>>>(
                thrust::raw_pointer_cast(d_input.data()),
                thrust::raw_pointer_cast(d_max.data()),
                thrust::raw_pointer_cast(d_sum.data()),
                static_cast<int>(n));

            softmax_normalize<<<grid_size, block_size, 0, launch.get_stream()>>>(
                thrust::raw_pointer_cast(d_input.data()),
                thrust::raw_pointer_cast(d_max.data()),
                thrust::raw_pointer_cast(d_sum.data()),
                thrust::raw_pointer_cast(d_output.data()),
                static_cast<int>(n));
        });
}

NVBENCH_BENCH(softmax_gpu_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});