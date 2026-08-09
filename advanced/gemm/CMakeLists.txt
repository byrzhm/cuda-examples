add_executable(gemm gemm.cu)
target_link_libraries(gemm nvbench::main)

add_executable(gemm_check gemm_check.cu)