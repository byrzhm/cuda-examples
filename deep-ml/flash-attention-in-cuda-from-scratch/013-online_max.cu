__device__ float online_max(float old_max, float new_val) {
    // return the running max of old_max and new_val
    return fmaxf(old_max, new_val);
}
