__device__ float update_running_sum (float old_sum, float correction, float block_sum) {
    // combine the rescaled old sum with the new block sum
    return old_sum * correction + block_sum;
}
