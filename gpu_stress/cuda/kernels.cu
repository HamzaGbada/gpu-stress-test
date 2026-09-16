// CUDA C kernels for the zero-dependency GPU stress pipeline.
//
// Compiled ahead of time to PTX (see build_ptx.sh) and loaded through the
// CUDA driver API (libcuda.so) with ctypes -- no toolkit, no PyTorch and no
// numba are needed at runtime, only the NVIDIA driver.
//
//   kernels_sm70.ptx : everything except the tensor-core kernel (sm_70+)
//   kernels_sm80.ptx : mma_burn, needs mma.sync.m16n8k16 (sm_80+)

// No headers: fp16 is done with inline PTX so NVRTC needs no include path.

// ---------------------------------------------------------------------------
// fp16x2 helpers (packed pair of halves in one 32-bit register)
// ---------------------------------------------------------------------------
__device__ __forceinline__ unsigned f2_to_h2(float lo, float hi)
{
    unsigned r;
    asm("{.reg .f16 l, h;\n"
        " cvt.rn.f16.f32 l, %1;\n"
        " cvt.rn.f16.f32 h, %2;\n"
        " mov.b32 %0, {l, h};}\n" : "=r"(r) : "f"(lo), "f"(hi));
    return r;
}

__device__ __forceinline__ float h2_sum(unsigned h)
{
    float lo, hi;
    asm("{.reg .f16 l, h;\n"
        " mov.b32 {l, h}, %2;\n"
        " cvt.f32.f16 %0, l;\n"
        " cvt.f32.f16 %1, h;}\n" : "=f"(lo), "=f"(hi) : "r"(h));
    return lo + hi;
}

__device__ __forceinline__ unsigned hfma2(unsigned a, unsigned b, unsigned c)
{
    unsigned d;
    asm("fma.rn.f16x2 %0, %1, %2, %3;" : "=r"(d) : "r"(a), "r"(b), "r"(c));
    return d;
}

// ---------------------------------------------------------------------------
// FP32 FMA burn: 8 independent dependency chains per thread so the scheduler
// can keep the FMA pipes full. Every chain converges to exactly 1.0f, which
// makes the output verifiable: any bit-flip in an ALU shows up as out != 1.
// FLOPs = 2 * 8 * iters per thread.
// ---------------------------------------------------------------------------
extern "C" __global__ void fma_burn(float *out, int iters, float seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float k = 0.999f, c = 0.001f;
    float a0 = seed + tid * 1e-6f, a1 = a0 + 0.1f, a2 = a0 + 0.2f, a3 = a0 + 0.3f;
    float a4 = a0 + 0.4f, a5 = a0 + 0.5f, a6 = a0 + 0.6f, a7 = a0 + 0.7f;
    for (int i = 0; i < iters; ++i) {
        a0 = fmaf(a0, k, c); a1 = fmaf(a1, k, c); a2 = fmaf(a2, k, c); a3 = fmaf(a3, k, c);
        a4 = fmaf(a4, k, c); a5 = fmaf(a5, k, c); a6 = fmaf(a6, k, c); a7 = fmaf(a7, k, c);
    }
    out[tid] = (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7) * 0.125f;
}

// ---------------------------------------------------------------------------
// FP16x2 burn using packed half2 FMAs (2 flops per lane, 2 lanes per op).
// Same convergence trick (a = 0.9a + 0.1 -> 1.0), output converted to fp32.
// FLOPs = 2 * 2 * 8 * iters per thread.
// ---------------------------------------------------------------------------
extern "C" __global__ void hfma2_burn(float *out, int iters, float seed)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned k = f2_to_h2(0.9f, 0.9f);
    const unsigned c = f2_to_h2(0.1f, 0.1f);
    const float s = seed + (tid & 1023) * 1e-3f;
    unsigned a0 = f2_to_h2(s, s + 0.1f), a1 = f2_to_h2(s + 0.2f, s + 0.3f);
    unsigned a2 = f2_to_h2(s + 0.4f, s + 0.5f), a3 = f2_to_h2(s + 0.6f, s + 0.7f);
    unsigned a4 = a0, a5 = a1, a6 = a2, a7 = a3;
    for (int i = 0; i < iters; ++i) {
        a0 = hfma2(a0, k, c); a1 = hfma2(a1, k, c); a2 = hfma2(a2, k, c); a3 = hfma2(a3, k, c);
        a4 = hfma2(a4, k, c); a5 = hfma2(a5, k, c); a6 = hfma2(a6, k, c); a7 = hfma2(a7, k, c);
    }
    out[tid] = (h2_sum(a0) + h2_sum(a1) + h2_sum(a2) + h2_sum(a3)
              + h2_sum(a4) + h2_sum(a5) + h2_sum(a6) + h2_sum(a7)) * 0.0625f;
}

// ---------------------------------------------------------------------------
// Register-tiled SGEMM: C[M,N] = A[M,K] * B[K,N], row-major.
// 64x64 block tile, BK=16, 256 threads each computing a 4x4 micro-tile.
// Not cuBLAS, but reaches a decent fraction of FP32 peak and is exactly
// verifiable on the host when A/B hold small integers.
// ---------------------------------------------------------------------------
#define BM 64
#define BN 64
#define BK 16
#define TM 4
#define TN 4
extern "C" __global__ void sgemm_tiled(const float *__restrict__ A, const float *__restrict__ B,
                                       float *__restrict__ C, int M, int N, int K)
{
    __shared__ float As[BK][BM + 4];   // stored transposed: As[k][m]
    __shared__ float Bs[BK][BN];
    const int tid = threadIdx.x;               // 0..255
    const int tx = tid % (BN / TN), ty = tid / (BN / TN);
    const int row0 = blockIdx.y * BM, col0 = blockIdx.x * BN;
    float acc[TM][TN];
#pragma unroll
    for (int i = 0; i < TM; ++i)
#pragma unroll
        for (int j = 0; j < TN; ++j) acc[i][j] = 0.f;

    for (int t = 0; t < K; t += BK) {
        // 64x16 tile of A: 4 elements per thread, consecutive threads -> consecutive k
        for (int i = tid; i < BM * BK; i += 256) {
            const int m = i / BK, k = i % BK;
            const int gm = row0 + m, gk = t + k;
            As[k][m] = (gm < M && gk < K) ? A[(size_t)gm * K + gk] : 0.f;
        }
        // 16x64 tile of B: consecutive threads -> consecutive n (coalesced)
        for (int i = tid; i < BK * BN; i += 256) {
            const int k = i / BN, n = i % BN;
            const int gk = t + k, gn = col0 + n;
            Bs[k][n] = (gk < K && gn < N) ? B[(size_t)gk * N + gn] : 0.f;
        }
        __syncthreads();
#pragma unroll
        for (int k = 0; k < BK; ++k) {
            float ra[TM], rb[TN];
#pragma unroll
            for (int i = 0; i < TM; ++i) ra[i] = As[k][ty * TM + i];
#pragma unroll
            for (int j = 0; j < TN; ++j) rb[j] = Bs[k][tx * TN + j];
#pragma unroll
            for (int i = 0; i < TM; ++i)
#pragma unroll
                for (int j = 0; j < TN; ++j) acc[i][j] = fmaf(ra[i], rb[j], acc[i][j]);
        }
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < TM; ++i) {
        const int gm = row0 + ty * TM + i;
        if (gm >= M) continue;
#pragma unroll
        for (int j = 0; j < TN; ++j) {
            const int gn = col0 + tx * TN + j;
            if (gn < N) C[(size_t)gm * N + gn] = acc[i][j];
        }
    }
}

// Deterministic small-integer fill so matmul results are exactly checkable:
// value = ((i * mul + add) mod range) - range/2   in  [-range/2, range/2)
extern "C" __global__ void fill_pattern_f32(float *buf, unsigned long long n, unsigned mul, unsigned add, unsigned range)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        unsigned v = ((unsigned)i * mul + add) % range;
        buf[i] = (float)((int)v - (int)(range / 2));
    }
}

// ---------------------------------------------------------------------------
// VRAM integrity test (mini memtest): write pattern ^ index, read back, count
// mismatches with an atomic counter.
// ---------------------------------------------------------------------------
extern "C" __global__ void mem_fill(unsigned *buf, unsigned long long n, unsigned pattern)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        buf[i] = pattern ^ (unsigned)i;
}

extern "C" __global__ void mem_check(const unsigned *buf, unsigned long long n, unsigned pattern, unsigned *errors)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned local = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        local += (buf[i] != (pattern ^ (unsigned)i));
    if (local) atomicAdd(errors, local);
}

// ---------------------------------------------------------------------------
// Vectorised device-to-device copy for bandwidth measurement (float4 = 16 B).
// ---------------------------------------------------------------------------
extern "C" __global__ void copy_f4(const float4 *__restrict__ src, float4 *__restrict__ dst, unsigned long long n4)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride)
        dst[i] = src[i];
}

// ---------------------------------------------------------------------------
// Tensor-core burn (sm_80+): each warp runs 4 independent
// mma.sync.m16n8k16 f16->f32 chains. With A = B = 1.0h every MMA adds K=16 to
// every accumulator, so after `iters` steps each thread's 4 chains x 4 regs sum
// to exactly 256*iters (exact in fp32 while 256*iters < 2^24).
// FLOPs = 4 chains * 2*16*8*16 = 16384 per warp per iteration.
// ---------------------------------------------------------------------------
#if __CUDA_ARCH__ >= 800
extern "C" __global__ void mma_burn(float *out, int iters)
{
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const unsigned one2 = 0x3c003c00u;  // two fp16 1.0
    float c[4][4] = {{0}};
    for (int i = 0; i < iters; ++i) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            asm volatile(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+f"(c[j][0]), "+f"(c[j][1]), "+f"(c[j][2]), "+f"(c[j][3])
                : "r"(one2), "r"(one2), "r"(one2), "r"(one2), "r"(one2), "r"(one2));
        }
    }
    float s = 0.f;
#pragma unroll
    for (int j = 0; j < 4; ++j) s += c[j][0] + c[j][1] + c[j][2] + c[j][3];
    out[tid] = s;
}
#endif
