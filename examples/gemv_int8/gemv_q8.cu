// gemv_q8.cu - the CUDA original: y[c][n] = sx[c] * sw[n] * sum_k W[n][k] x[c][k] for int8 weights W (n_out rows of n_in bytes, one
// fp32 scale per row), int8 activations x (ncols rows, one fp32 scale each). One warp per output row, each lane takes 4 bytes per
// step (dp4a), the 32 lane sums meet in a butterfly. The port is gemv_q8.sycl.cpp.
#include <cuda_runtime.h>

template <int NCOLS>
__global__ void gemv_q8_kernel(const int8_t* __restrict__ W, const float* __restrict__ sw, const int8_t* __restrict__ x,
                               const float* __restrict__ sx, float* __restrict__ y, int n_in, int n_out) {
    const int row = blockIdx.x * 4 + threadIdx.y;   // 4 warps per block
    if (row >= n_out) return;
    const int lane = threadIdx.x;
    const int* w4 = reinterpret_cast<const int*>(W + (size_t) row * n_in);
    int acc[NCOLS] = {};
    for (int k = lane; k < n_in / 4; k += 32) {
        const int w = w4[k];
#pragma unroll
        for (int c = 0; c < NCOLS; ++c) acc[c] = __dp4a(w, reinterpret_cast<const int*>(x + (size_t) c * n_in)[k], acc[c]);
    }
#pragma unroll
    for (int c = 0; c < NCOLS; ++c) {
        int v = acc[c];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffff, v, o);
        if (lane == 0) y[(size_t) c * n_out + row] = sx[c] * sw[row] * (float) v;
    }
}

template <int NCOLS>
void gemv_q8(const int8_t* W, const float* sw, const int8_t* x, const float* sx, float* y, int n_in, int n_out, cudaStream_t s) {
    if (n_in <= 0 || n_in > 131071 || n_in % 4 || n_out <= 0) return;
    gemv_q8_kernel<NCOLS><<<(n_out + 3) / 4, dim3(32, 4), 0, s>>>(W, sw, x, sx, y, n_in, n_out);
}
