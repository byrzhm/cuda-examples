__device__ float correction_factor (float old_max, float new_max) {
    // return the scalar used to rescale running statistics
    return expf(old_max - new_max);
}
