# sum
add_executable(sum sum.cu)
target_link_libraries(sum nvbench::main)

add_executable(sum_check sum_check.cu)

# softmax
add_executable(softmax softmax.cu)
target_link_libraries(softmax nvbench::main)

add_executable(softmax_check softmax_check.cu)