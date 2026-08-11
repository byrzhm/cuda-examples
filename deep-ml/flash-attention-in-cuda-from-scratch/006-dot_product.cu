// 可以优化
__device__ float dot_product(const float* a, const float* b, int n) {
    float res = 0.0f;
    for (int i = 0; i < n; ++i) {
        res += a[i] * b[i];
    }
    return res;
}
