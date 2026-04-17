add_executable(sum sum.cu)
target_link_libraries(sum nvbench::main)

add_executable(sum_check sum_check.cu)