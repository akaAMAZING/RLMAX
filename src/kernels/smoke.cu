// Toolchain smoke kernel: proves nvcc, the fat-binary embed and the driver-API launch path end to end.
// It is not physics and is removed once the first physics kernel exists.

extern "C" __global__ void smoke_axpy(float a, const float* __restrict__ x, float* __restrict__ y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = fmaf(a, x[i], y[i]);
}
