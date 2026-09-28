// CUDA C kernels for the zero-dependency GPU stress pipeline.
//
// Compiled ahead of time to PTX (see nvrtc.py) and loaded through the CUDA
// driver API (libcuda.so) with ctypes -- no toolkit, no PyTorch and no numba
// are needed at runtime, only the NVIDIA driver.
//
//   kernels_sm70.ptx : everything except the tensor-core kernel (sm_70+)
//   kernels_sm80.ptx : adds mma_burn, needs mma.sync.m16n8k16 (sm_80+)
//
// No headers are included: fp16 is done with inline PTX and every other
// builtin (fmaf, atomics, warp intrinsics, vector types) is pre-declared by
// NVRTC, so the shipped PTX can be rebuilt without any CUDA include path.

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
// VRAM integrity test (mini memtest).
//
// Writes `pattern ^ word_index` over the buffer and reads it back, counting
// mismatches with an atomic counter. Both kernels move 16 B per thread per
// step (uint4) because scalar 4 B accesses leave most of the memory pipeline
// idle on wide-bus and unified-memory parts.
//
// `n4` counts uint4 elements. The word index is truncated to 32 bits, so the
// data pattern repeats every 16 GiB - fill and check agree, and the value
// still depends on the address within each window.
// ---------------------------------------------------------------------------
extern "C" __global__ void mem_fill(uint4 *buf, unsigned long long n4, unsigned pattern)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const unsigned w = (unsigned)i << 2;
        uint4 v;
        v.x = pattern ^ w;
        v.y = pattern ^ (w + 1u);
        v.z = pattern ^ (w + 2u);
        v.w = pattern ^ (w + 3u);
        buf[i] = v;
    }
}

extern "C" __global__ void mem_check(const uint4 *__restrict__ buf, unsigned long long n4,
                                     unsigned pattern, unsigned *errors)
{
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned local = 0;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; i < n4; i += stride) {
        const unsigned w = (unsigned)i << 2;
        const uint4 v = buf[i];
        local += (v.x != (pattern ^ w));
        local += (v.y != (pattern ^ (w + 1u)));
        local += (v.z != (pattern ^ (w + 2u)));
        local += (v.w != (pattern ^ (w + 3u)));
    }
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
// Reductions used to verify the physics steps. Accumulation is in fp64 so the
// invariant check is limited by the simulation, not by the reduction.
// Both write one group of doubles per block; the host sums the blocks.
// Launch with exactly RED_BLOCK threads per block.
// ---------------------------------------------------------------------------
#define RED_BLOCK 256
#define DBL_BIG 1.7976931348623157e308

extern "C" __global__ void reduce_stats_f32(const float *__restrict__ in, unsigned long long n, double *out)
{
    __shared__ double ssum[RED_BLOCK];
    __shared__ double smin[RED_BLOCK];
    __shared__ double smax[RED_BLOCK];
    const unsigned t = threadIdx.x;
    const unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    double sum = 0.0, mn = DBL_BIG, mx = -DBL_BIG;
    for (unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + t; i < n; i += stride) {
        const double v = (double)in[i];
        sum += v;
        mn = v < mn ? v : mn;
        mx = v > mx ? v : mx;
    }
    ssum[t] = sum; smin[t] = mn; smax[t] = mx;
    __syncthreads();
    for (unsigned s = RED_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) {
            ssum[t] += ssum[t + s];
            smin[t] = smin[t + s] < smin[t] ? smin[t + s] : smin[t];
            smax[t] = smax[t + s] > smax[t] ? smax[t + s] : smax[t];
        }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x * 3 + 0] = ssum[0];
        out[blockIdx.x * 3 + 1] = smin[0];
        out[blockIdx.x * 3 + 2] = smax[0];
    }
}

// Total momentum (m*v) and momentum magnitude scale (m*|v|) of an N-body state.
extern "C" __global__ void nbody_reduce(const float4 *__restrict__ vel, unsigned n, double *out)
{
    __shared__ double sx[RED_BLOCK], sy[RED_BLOCK], sz[RED_BLOCK], sa[RED_BLOCK];
    const unsigned t = threadIdx.x;
    const unsigned stride = gridDim.x * blockDim.x;
    double px = 0.0, py = 0.0, pz = 0.0, pa = 0.0;
    for (unsigned i = blockIdx.x * blockDim.x + t; i < n; i += stride) {
        const float4 v = vel[i];
        const double m = (double)v.w;
        px += m * (double)v.x;
        py += m * (double)v.y;
        pz += m * (double)v.z;
        pa += m * (double)sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    }
    sx[t] = px; sy[t] = py; sz[t] = pz; sa[t] = pa;
    __syncthreads();
    for (unsigned s = RED_BLOCK / 2; s > 0; s >>= 1) {
        if (t < s) { sx[t] += sx[t + s]; sy[t] += sy[t + s]; sz[t] += sz[t + s]; sa[t] += sa[t + s]; }
        __syncthreads();
    }
    if (t == 0) {
        out[blockIdx.x * 4 + 0] = sx[0];
        out[blockIdx.x * 4 + 1] = sy[0];
        out[blockIdx.x * 4 + 2] = sz[0];
        out[blockIdx.x * 4 + 3] = sa[0];
    }
}

// ---------------------------------------------------------------------------
// Physics 1: direct N-body gravity (O(N^2), shared-memory tiled).
//
// pos/vel are float4: xyz + mass in .w for pos, xyz + mass in .w for vel.
// Leapfrog-ish: a = sum_j m_j * r_ij / (|r_ij|^2 + eps^2)^{3/2}, then
// v += a*dt, p += v*dt. Self-interaction contributes exactly zero (r = 0).
//
// Physical invariant: Newton's third law makes the forces antisymmetric, so
// total momentum is conserved. The host starts from zero net momentum and
// checks it stays there - a silent ALU fault breaks the symmetry immediately.
//
// n must be a multiple of NB_TILE and grid = n / NB_TILE (no bounds checks).
// FLOPs ~ 20 per interaction (the usual N-body convention).
// ---------------------------------------------------------------------------
#define NB_TILE 256
extern "C" __global__ void nbody_step(const float4 *__restrict__ pos, const float4 *__restrict__ vel,
                                      float4 *__restrict__ out_pos, float4 *__restrict__ out_vel,
                                      int n, float dt, float eps2)
{
    __shared__ float4 sh[NB_TILE];
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const float4 p = pos[tid];
    float ax = 0.f, ay = 0.f, az = 0.f;

    for (int tile = 0; tile < n; tile += NB_TILE) {
        sh[threadIdx.x] = pos[tile + threadIdx.x];
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < NB_TILE; ++j) {
            const float dx = sh[j].x - p.x;
            const float dy = sh[j].y - p.y;
            const float dz = sh[j].z - p.z;
            const float d2 = fmaf(dx, dx, fmaf(dy, dy, fmaf(dz, dz, eps2)));
            const float inv = rsqrtf(d2);
            const float s = sh[j].w * inv * inv * inv;
            ax = fmaf(dx, s, ax);
            ay = fmaf(dy, s, ay);
            az = fmaf(dz, s, az);
        }
        __syncthreads();
    }

    float4 v = vel[tid];
    v.x = fmaf(ax, dt, v.x);
    v.y = fmaf(ay, dt, v.y);
    v.z = fmaf(az, dt, v.z);
    out_vel[tid] = v;

    float4 np = p;
    np.x = fmaf(v.x, dt, p.x);
    np.y = fmaf(v.y, dt, p.y);
    np.z = fmaf(v.z, dt, p.z);
    out_pos[tid] = np;
}

// ---------------------------------------------------------------------------
// Physics 2: 2D heat diffusion, 5-point Jacobi stencil, periodic boundaries.
//
//   u' = u + alpha * (uN + uS + uE + uW - 4u)
//
// Two invariants the host checks, both exact in real arithmetic:
//   * conservation: with periodic BCs the total heat sum(u) is unchanged;
//   * maximum principle: for 0 <= alpha <= 0.25 the update is a convex
//     combination, so u' can never leave the initial [min, max] range.
// Either one breaking means the GPU computed something wrong.
//
// Width/height are powers of two so the wrap-around is a mask, not a modulo.
// Memory-bound: ~5 loads + 1 store per cell, ~6 flops.
// ---------------------------------------------------------------------------
extern "C" __global__ void heat_step(const float *__restrict__ u, float *__restrict__ un,
                                     int w, int h, float alpha)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= w || y >= h) return;
    const int wm = w - 1, hm = h - 1;          // w, h are powers of two
    const int row = y * w;
    const float c = u[row + x];
    const float e = u[row + ((x + 1) & wm)];
    const float we = u[row + ((x - 1) & wm)];
    const float s = u[(((y + 1) & hm)) * w + x];
    const float nr = u[(((y - 1) & hm)) * w + x];
    un[row + x] = fmaf(alpha, (e + we + s + nr) - 4.0f * c, c);
}

// ---------------------------------------------------------------------------
// Hardware edge cases: IEEE-754 corner values, integer/64-bit arithmetic,
// atomics, warp intrinsics and shared-memory sync. Every check has a single
// right answer, so a set bit means the GPU (or its clocks) is misbehaving.
//
// Inputs come from memory so nothing can be constant-folded at compile time.
// Launch with exactly 256 threads per block.
// ---------------------------------------------------------------------------
#define EC_DENORMAL_FLUSHED 0x0001u
#define EC_NAN_COMPARE      0x0002u
#define EC_INF_ARITH        0x0004u
#define EC_INT_OPS          0x0008u
#define EC_SHFL             0x0010u
#define EC_BALLOT           0x0020u
#define EC_SHARED_REDUCE    0x0040u
#define EC_FMA_NOT_FUSED    0x0080u
#define EC_SQRT_SPECIAL     0x0100u
#define EC_ROUNDING         0x0200u
#define EC_INT64            0x0400u

extern "C" __global__ void edge_cases(const float *__restrict__ in, unsigned *flags, unsigned *counters, int seed)
{
    __shared__ unsigned sred[256];
    const unsigned tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned m = 0;

    const float den = in[0];      // 1.4e-45f, the smallest denormal (bits 0x1)
    const float zero = in[1];     // 0.0f
    const float inf = in[2];      // +inf
    const float qnan = in[3];     // NaN
    const float a = in[6];        // 1 + 2^-23
    const float b = in[7];        // 1 - 2^-24
    const float negone = in[8];   // -1.0f
    const float two = in[9];      // 2.0f

    // -- denormals: den * 2 must stay denormal (bits 0x2), not flush to zero --
    if (__float_as_uint(den * two) != 0x00000002u) m |= EC_DENORMAL_FLUSHED;

    // -- NaN comparisons: every ordered compare with NaN is false --
    if ((qnan == qnan) || !(qnan != qnan) || (qnan < two) || (qnan >= two)) m |= EC_NAN_COMPARE;

    // -- infinities --
    if (__float_as_uint(1.0f / zero) != 0x7F800000u) m |= EC_INF_ARITH;
    if ((__float_as_uint(inf - inf) & 0x7FFFFFFFu) <= 0x7F800000u) m |= EC_INF_ARITH;  // must be NaN
    if (__float_as_uint(inf + two) != 0x7F800000u) m |= EC_INF_ARITH;
    if (__float_as_uint(negone * inf) != 0xFF800000u) m |= EC_INF_ARITH;

    // -- rounding: (1+2^-23)*(1-2^-24) rounds to exactly 1.0, and 0.1f+0.2f --
    if (__fmul_rn(a, b) != 1.0f) m |= EC_ROUNDING;
    if (__float_as_uint(__fadd_rn(in[4], in[5])) != 0x3E99999Au) m |= EC_ROUNDING;

    // -- FMA must be fused: the product's tail survives the add --
    if (fmaf(a, b, negone) == 0.0f) m |= EC_FMA_NOT_FUSED;

    // -- sqrt of special values --
    if ((__float_as_uint(sqrtf(negone)) & 0x7FFFFFFFu) <= 0x7F800000u) m |= EC_SQRT_SPECIAL;
    if (__float_as_uint(sqrtf(zero)) != 0u) m |= EC_SQRT_SPECIAL;
    if (__float_as_uint(sqrtf(inf)) != 0x7F800000u) m |= EC_SQRT_SPECIAL;

    // -- 32-bit integer intrinsics with known answers --
    if (__popc(0xDEADBEEFu) != 24) m |= EC_INT_OPS;
    if (__clz(0x00F00000u) != 8) m |= EC_INT_OPS;
    if (__umulhi(0xFFFFFFFFu, 0xFFFFFFFFu) != 0xFFFFFFFEu) m |= EC_INT_OPS;
    const unsigned d = (unsigned)(seed & 63) + 4u;          // runtime, cannot be folded
    if ((d * 7u) / d != 7u) m |= EC_INT_OPS;
    if ((d * 7u + 3u) % d != 3u % d) m |= EC_INT_OPS;

    // -- 64-bit arithmetic --
    const unsigned long long big = 0x0123456789ABCDEFull;
    const unsigned long long q = big * (unsigned long long)d;
    if (q / (unsigned long long)d != big) m |= EC_INT64;
    if (((q << 3) >> 3) != (q & 0x1FFFFFFFFFFFFFFFull)) m |= EC_INT64;

    // -- warp shuffle butterfly: every lane ends with sum(0..31) = 496 --
    {
        float v = (float)(threadIdx.x & 31u);
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xFFFFFFFFu, v, off, 32);
        if (v != 496.0f) m |= EC_SHFL;
    }

    // -- ballot: even lanes set -> 0x55555555 --
    if (__ballot_sync(0xFFFFFFFFu, (threadIdx.x & 1u) == 0u) != 0x55555555u) m |= EC_BALLOT;

    // -- shared memory + __syncthreads tree reduction: sum(0..255) = 32640 --
    sred[threadIdx.x] = threadIdx.x;
    __syncthreads();
    for (unsigned s = 128; s > 0; s >>= 1) {
        if (threadIdx.x < s) sred[threadIdx.x] += sred[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0 && sred[0] != 32640u) m |= EC_SHARED_REDUCE;

    // -- atomics: the host checks the exact expected totals --
    atomicAdd(&counters[0], 1u);
    atomicMax(&counters[1], tid);
    atomicXor(&counters[3], tid);
    // One CAS-loop increment per warp: a compare-and-swap retry loop on a single
    // address from every thread of a 100k-thread grid serialises for seconds.
    if ((threadIdx.x & 31u) == 0u) {
        unsigned old = counters[2], assumed;
        do {
            assumed = old;
            old = atomicCAS(&counters[2], assumed, assumed + 1u);
        } while (assumed != old);
    }

    if (m) atomicOr(flags, m);
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
