add_executable(transpose transpose.cu)
target_link_libraries(transpose nvbench::main)

add_executable(transpose_check transpose_check.cu)