// silu_mul.cu - the CUDA original: out[t][i] = silu(x[t][i]) * x[t][d + i] for d columns of 2d-wide rows (the gated activation of
// an MLP). One block per token row, float4 per thread, grid-stride over the row. The port is silu_mul.sycl.cpp.
#include <cuda_runtime.h>

__device__ __forceinline__ float silu(float x) { return x / (1.0f + __expf(-x)); }

__global__ void silu_mul_kernel(const float* __restrict__ x, float* __restrict__ out, int d) {
    const int t = blockIdx.x;
    const float* a = x + (size_t) t * 2 * d;
    const float* b = a + d;
    float* o = out + (size_t) t * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) o[i] = silu(a[i]) * b[i];
}

void silu_mul(const float* x, float* out, int tokens, int d, cudaStream_t s) {
    if (tokens <= 0 || d <= 0) return;
    silu_mul_kernel<<<tokens, 512, 0, s>>>(x, out, d);
}
