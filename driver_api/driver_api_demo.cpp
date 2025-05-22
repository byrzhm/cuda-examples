#include <cuda.h>
#include <iostream>
#include <vector>

#define CHECK_CUDA(call)                                                      \
    do {                                                                      \
        CUresult err = call;                                                  \
        if (err != CUDA_SUCCESS) {                                            \
            const char *msg;                                                  \
            cuGetErrorName(err, &msg);                                        \
            std::cerr << "CUDA error: " << msg << " at line " << __LINE__;   \
            exit(1);                                                          \
        }                                                                     \
    } while (0)

int main() {
    int N = 1024;
    size_t size = N * sizeof(float);
    std::vector<float> host_data(N);
    for (int i = 0; i < N; ++i) host_data[i] = float(i);

    // Initialize the driver
    CHECK_CUDA(cuInit(0));

    // Get device and create context
    CUdevice device;
    CHECK_CUDA(cuDeviceGet(&device, 0));
    CUcontext context;
    CHECK_CUDA(cuCtxCreate(&context, 0, device));

    // Load PTX module and get kernel
    CUmodule module;
    CHECK_CUDA(cuModuleLoad(&module, "square_kernel.ptx"));
    CUfunction kernel;
    CHECK_CUDA(cuModuleGetFunction(&kernel, module, "square"));

    // Allocate and copy memory to GPU
    CUdeviceptr d_data;
    CHECK_CUDA(cuMemAlloc(&d_data, size));
    CHECK_CUDA(cuMemcpyHtoD(d_data, host_data.data(), size));

    // Launch kernel
    void *args[] = { &d_data, &N };
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    CHECK_CUDA(cuLaunchKernel(kernel,
                              blocks, 1, 1,       // grid dim
                              threads, 1, 1,      // block dim
                              0, nullptr,         // shared mem and stream
                              args, nullptr));    // args and extra
    CHECK_CUDA(cuCtxSynchronize());

    // Copy result back
    CHECK_CUDA(cuMemcpyDtoH(host_data.data(), d_data, size));

    // Print some results
    for (int i = 0; i < 10; ++i)
        std::cout << "host_data[" << i << "] = " << host_data[i] << "\n";

    // Cleanup
    cuMemFree(d_data);
    cuModuleUnload(module);
    cuCtxDestroy(context);

    return 0;
}
