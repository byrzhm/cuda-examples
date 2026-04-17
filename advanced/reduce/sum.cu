#include <nvbench/nvbench.cuh>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

__global__ void sum_naive(const float* input, float* output, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) atomicAdd(output, input[idx]);
}

void sum_naive_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(1, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);   // Read n input elements
    state.add_global_memory_writes<float>(1);   // Write 1 output (the sum)

    state.exec([&input, &output, n, block_size, grid_size](nvbench::launch& launch) {
        sum_naive<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));
    });
}

NVBENCH_BENCH(sum_naive_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});



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


void sum_v0_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(grid_size, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(grid_size);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size](nvbench::launch& launch) {
        sum_v0<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()));

        // sum in host
        thrust::host_vector<float> host_output = output;
        float final_sum = 0.0f;
        for (int i = 0; i < grid_size; ++i) {
            final_sum += host_output[i];
        }
    });
}

NVBENCH_BENCH(sum_v0_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});




__global__ void sum_v1(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ float sdata[256];

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

void sum_v1_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(grid_size, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(grid_size);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size](nvbench::launch& launch) {
        sum_v1<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));

        // sum in host
        thrust::host_vector<float> host_output = output;
        float final_sum = 0.0f;
        for (int i = 0; i < grid_size; ++i) {
            final_sum += host_output[i];
        }
    });
}

NVBENCH_BENCH(sum_v1_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});


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

void sum_v2_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(grid_size, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(grid_size);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size, shared_mem_size](nvbench::launch& launch) {
        sum_v2<<<grid_size, block_size, shared_mem_size, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));

        // sum in host
        thrust::host_vector<float> host_output = output;
        float final_sum = 0.0f;
        for (int i = 0; i < grid_size; ++i) {
            final_sum += host_output[i];
        }
    });
}

NVBENCH_BENCH(sum_v2_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});


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


void sum_v3_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);
    const size_t shared_mem_size = block_size * sizeof(float);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(1, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(1);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size, shared_mem_size](nvbench::launch& launch) {
        sum_v3<<<grid_size, block_size, shared_mem_size, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));

        thrust::host_vector<float> host_output = output;
        float final_sum = host_output[0];
    });
}

NVBENCH_BENCH(sum_v3_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});


// Warp shuffle version
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

void sum_v4_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    const int block_size = 256;
    const int grid_size = static_cast<int>((n + block_size - 1) / block_size);

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(1, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(1);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size](nvbench::launch& launch) {
        sum_v4<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));

        thrust::host_vector<float> host_output = output;
        float final_sum = host_output[0];
    });
}

NVBENCH_BENCH(sum_v4_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});


__global__ void sum_v5(const float* input, float* output, int N) {
    int tid = threadIdx.x;
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    int warp_id = tid / 32;
    int lane_id = tid % 32;
    int num_warps = blockDim.x / 32;

    __shared__ float sdata[32];

    // Load data with boundary check
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

void sum_v5_bench(nvbench::state& state) {
    const std::size_t n = static_cast<std::size_t>(state.get_int64("N"));

    auto ceil_div = [](int x, int y) {
        return (x + y - 1) / y;
    };

    const int block_size = 256;
    const int grid_size = static_cast<int>(ceil_div(ceil_div(n, block_size), 4));

    thrust::device_vector<float> input(n, 1.0f);
    thrust::device_vector<float> output(1, 0.0f);

    state.add_element_count(n);
    state.add_global_memory_reads<float>(n);
    state.add_global_memory_writes<float>(1);

    state.exec(nvbench::exec_tag::sync, [&input, &output, n, block_size, grid_size](nvbench::launch& launch) {
        sum_v5<<<grid_size, block_size, 0, launch.get_stream()>>>(
            thrust::raw_pointer_cast(input.data()),
            thrust::raw_pointer_cast(output.data()),
            static_cast<int>(n));

        thrust::host_vector<float> host_output = output;
        float final_sum = host_output[0];
    });
}

NVBENCH_BENCH(sum_v5_bench)
    .add_int64_axis("N", {1024, 4096, 16384, 65536, 262144, 1048576, 4194304, 16777216});