__device__ float warp_reduce_max (float val) {
    // implement warp-level max reduction using shuffle intrinsics
    int mask = __activemask ();
    int active = __popc(mask);
    int max_off = 1;
    while (max_off < active) max_off <<= 1;
    max_off >>= 1;
    for (int off = max_off; off > 0; off >>= 1)
        val = fmaxf (val, __shfl_down_sync (mask, val, off));
    val = __shfl_sync (mask, val, 0);
    return val;
}
