// SPDX-License-Identifier: Apache-2.0
// Copyright (c) 2026 RL-Kernel Contributors
//
// Hopper (SM90) fused linear log-prob:
//
//     logp[n] = log_softmax(hidden[n] @ W^T + b)[target[n]]
//
// computed WITHOUT materializing the [N, V] logits. Each CTA owns a block of BM
// tokens and streams the vocabulary in BN-wide tiles: for every vocab tile it
//   1. computes the logit tile  logits[BM, BN] = H[BM, D] @ W[vtile, D]^T  with
//      WGMMA (K-loop over D), staging H/W tiles into shared memory via TMA,
//   2. stages the [BM, BN] fp32 logit tile into shared memory, and
//   3. folds it into a per-row online-softmax state (running max + sum) kept in
//      shared memory, capturing the target logit when its column lands in tile.
// After the vocab loop it writes logp = z_target - (max + log(sum)) and the row
// log-sum-exp (lse) for the backward pass.
//
// =========================================================================
// VALIDATION REQUIRED ON SM90.
// This file is written without an SM90 compiler/GPU in the loop. The numerically
// load-bearing, layout-sensitive pieces are:
//   (a) the WGMMA shared-memory matrix descriptors (LBO/SBO/swizzle), and
//   (b) the mapping from WGMMA fp32 accumulator registers -> (row, col).
// Both follow the PTX ISA "Asynchronous Warpgroup Level Matrix" section, but
// must be checked on hardware. The matmul is isolated in wgmma_m64n64k16 /
// store_acc_to_smem so it can be swapped for an mma.sync (m16n8k16) path if
// WGMMA needs debugging -- the surrounding TMA streaming + online softmax are
// independent of that choice.
// =========================================================================

#include "../utils/tma_utils.cuh"
#include <cuda_bf16.h>
#include <math_constants.h> // CUDART_INF_F (not pulled in transitively under CUDA 13)
#include <torch/extension.h>

namespace {

constexpr int BM = 64; // tokens per CTA  == WGMMA M
constexpr int BN = 64; // vocab per tile  == WGMMA N
constexpr int BK = 16; // contraction per WGMMA step == WGMMA K (bf16)
constexpr int WG_THREADS = 128; // one warpgroup (4 warps)
constexpr int ACC_REGS = (BM * BN) / WG_THREADS; // = 32 fp32 accumulators / thread

// --------------------------------------------------------------------------
// WGMMA helpers (SM90a). See PTX ISA: wgmma.mma_async / shared-memory matrix
// descriptor. Operands are streamed unswizzled (swizzle = 0); the descriptor
// encodes the smem start address plus the leading/stride byte offsets between
// 8x(BK) "core matrices".
// --------------------------------------------------------------------------
__device__ __forceinline__ uint64_t make_smem_desc(const void *smem_ptr, uint32_t lbo,
                                                    uint32_t sbo) {
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
    uint64_t desc = 0;
    desc |= (static_cast<uint64_t>(addr >> 4) & 0x3FFFULL);          // bits 0-13: start >> 4
    desc |= (static_cast<uint64_t>(lbo >> 4) & 0x3FFFULL) << 16;     // bits 16-29: LBO >> 4
    desc |= (static_cast<uint64_t>(sbo >> 4) & 0x3FFFULL) << 32;     // bits 32-45: SBO >> 4
    desc |= (static_cast<uint64_t>(0) << 62);                        // bits 62-63: swizzle = none
    return desc;
}

__device__ __forceinline__ void wgmma_fence() { asm volatile("wgmma.fence.sync.aligned;"); }
__device__ __forceinline__ void wgmma_commit() {
    asm volatile("wgmma.commit_group.sync.aligned;");
}
template <int N> __device__ __forceinline__ void wgmma_wait() {
    asm volatile("wgmma.wait_group.sync.aligned %0;" ::"n"(N));
}

// d += A @ B^T for one K=16 step. desc_a/desc_b are shared-memory descriptors.
// scale_d == 0 zeroes the accumulator first; 1 accumulates onto it.
__device__ __forceinline__ void wgmma_m64n64k16(float d[ACC_REGS], uint64_t desc_a,
                                                uint64_t desc_b, int scale_d) {
    // PTX form: {d}, desc_a, desc_b, scale-d (predicate), imm-scale-a, imm-scale-b,
    // imm-trans-a, imm-trans-b. scale-d is a predicate set from scale_d (0 = write,
    // 1 = accumulate). scaleA/scaleB = 1 (no negate); transA/transB = 0.
    asm volatile(
        "{\n"
        ".reg .pred p;\n"
        "setp.ne.b32 p, %34, 0;\n"
        "wgmma.mma_async.sync.aligned.m64n64k16.f32.bf16.bf16 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31}, "
        "%32, %33, p, 1, 1, 0, 0;\n"
        "}\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]), "+f"(d[4]), "+f"(d[5]), "+f"(d[6]),
          "+f"(d[7]), "+f"(d[8]), "+f"(d[9]), "+f"(d[10]), "+f"(d[11]), "+f"(d[12]), "+f"(d[13]),
          "+f"(d[14]), "+f"(d[15]), "+f"(d[16]), "+f"(d[17]), "+f"(d[18]), "+f"(d[19]), "+f"(d[20]),
          "+f"(d[21]), "+f"(d[22]), "+f"(d[23]), "+f"(d[24]), "+f"(d[25]), "+f"(d[26]), "+f"(d[27]),
          "+f"(d[28]), "+f"(d[29]), "+f"(d[30]), "+f"(d[31])
        : "l"(desc_a), "l"(desc_b), "r"(scale_d));
}

// Scatter the WGMMA fp32 accumulator to a row-major [BM][BN] shared-memory tile.
// m64nN .f32 layout (per warpgroup of 4 warps): warp w owns rows [16w, 16w+16);
// within a warp the N columns are tiled in 8-wide blocks, 4 regs per block:
//   reg[4j+0]->(lane/4,     8j+(lane%4)*2  )   reg[4j+1]->(lane/4,     8j+(lane%4)*2+1)
//   reg[4j+2]->(lane/4+8,   8j+(lane%4)*2  )   reg[4j+3]->(lane/4+8,   8j+(lane%4)*2+1)
__device__ __forceinline__ void store_acc_to_smem(const float d[ACC_REGS], float *s_logits) {
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int r0 = warp * 16 + lane / 4;
    const int r1 = r0 + 8;
    const int col = (lane % 4) * 2;
#pragma unroll
    for (int j = 0; j < BN / 8; ++j) {
        const int c = j * 8 + col;
        s_logits[r0 * BN + c + 0] = d[4 * j + 0];
        s_logits[r0 * BN + c + 1] = d[4 * j + 1];
        s_logits[r1 * BN + c + 0] = d[4 * j + 2];
        s_logits[r1 * BN + c + 1] = d[4 * j + 3];
    }
}

// --------------------------------------------------------------------------
// Kernel: one CTA (one warpgroup) per BM-token block.
// Shared memory layout (one buffer; double-buffering of A/B is a follow-up):
//   [ A tile BM*BK bf16 ][ B tile BN*BK bf16 ][ logits BM*BN f32 ]
//   [ row_max BM f32 ][ row_sum BM f32 ][ row_zt BM f32 ][ tma mbar 8B ]
// --------------------------------------------------------------------------
__global__ void fused_linear_logp_sm90_kernel(const __grid_constant__ CUtensorMap h_tmap,
                                              const __grid_constant__ CUtensorMap w_tmap,
                                              const int *__restrict__ target,
                                              const float *__restrict__ bias, // may be null
                                              float *__restrict__ out_logp,
                                              float *__restrict__ out_lse, int N, int D, int V) {
    const int tid = threadIdx.x;
    const int row_block = blockIdx.x;
    const int row_base = row_block * BM;
    const int num_rows = min(BM, N - row_base);
    const int kd = D / BK; // assumes D % BK == 0 (caller pads/validates)

    extern __shared__ __align__(1024) char smem[];
    nv_bfloat16 *sA = reinterpret_cast<nv_bfloat16 *>(smem);
    nv_bfloat16 *sB = reinterpret_cast<nv_bfloat16 *>(sA + BM * BK);
    float *sLogits = reinterpret_cast<float *>(sB + BN * BK);
    float *sMax = sLogits + BM * BN;
    float *sSum = sMax + BM;
    float *sZt = sSum + BM;
    const int tma_mbar = static_cast<int>(__cvta_generic_to_shared(sZt + BM));

    if (tid < num_rows) {
        sMax[tid] = -CUDART_INF_F;
        sSum[tid] = 0.0f;
        sZt[tid] = 0.0f;
    }
    if (tid == 0) {
        mbarrier_init(tma_mbar, 1);
        asm volatile("fence.mbarrier_init.release.cluster;");
    }
    __syncthreads();

    const int a_smem = static_cast<int>(__cvta_generic_to_shared(sA));
    const int b_smem = static_cast<int>(__cvta_generic_to_shared(sB));
    const uint32_t tile_bytes = (BM * BK + BN * BK) * sizeof(nv_bfloat16);
    int phase = 0;

    const int num_vtiles = (V + BN - 1) / BN;
    for (int vt = 0; vt < num_vtiles; ++vt) {
        const int col_base = vt * BN;

        float d[ACC_REGS];
#pragma unroll
        for (int i = 0; i < ACC_REGS; ++i)
            d[i] = 0.0f;

        // K-loop over the hidden dimension: stream H[BM,BK] and W[BN,BK] via TMA,
        // then issue one WGMMA per K step accumulating into d[].
        for (int k = 0; k < kd; ++k) {
            const int k_off = k * BK;
            if (tid == 0) {
                tma_2d_g2s(a_smem, &h_tmap, k_off, row_base, tma_mbar);
                tma_2d_g2s(b_smem, &w_tmap, k_off, col_base, tma_mbar);
                mbarrier_arrive_expect_tx(tma_mbar, tile_bytes);
            }
            mbarrier_wait(tma_mbar, phase);
            phase ^= 1;

            wgmma_fence();
            const uint64_t desc_a = make_smem_desc(sA, BK * sizeof(nv_bfloat16), 0);
            const uint64_t desc_b = make_smem_desc(sB, BK * sizeof(nv_bfloat16), 0);
            wgmma_m64n64k16(d, desc_a, desc_b, 1);
            wgmma_commit();
            wgmma_wait<0>();
            __syncthreads();
        }

        store_acc_to_smem(d, sLogits);
        __syncthreads();

        // Online softmax: one thread per row folds this tile's BN columns into
        // the running (max, sum) and captures the target logit if present.
        if (tid < num_rows) {
            const int r = tid;
            const int tgt = target[row_base + r];
            float tmax = -CUDART_INF_F;
            for (int c = 0; c < BN; ++c) {
                const int col = col_base + c;
                if (col >= V)
                    break;
                float val = sLogits[r * BN + c];
                if (bias != nullptr)
                    val += bias[col];
                tmax = fmaxf(tmax, val);
                if (col == tgt)
                    sZt[r] = val;
            }
            float tsum = 0.0f;
            for (int c = 0; c < BN; ++c) {
                const int col = col_base + c;
                if (col >= V)
                    break;
                float val = sLogits[r * BN + c];
                if (bias != nullptr)
                    val += bias[col];
                tsum += __expf(val - tmax);
            }
            float old_max = sMax[r];
            float new_max = fmaxf(old_max, tmax);
            sSum[r] = sSum[r] * __expf(old_max - new_max) + tsum * __expf(tmax - new_max);
            sMax[r] = new_max;
        }
        __syncthreads();
    }

    if (tid < num_rows) {
        const int r = tid;
        const float lse = sMax[r] + logf(sSum[r]);
        out_logp[row_base + r] = sZt[r] - lse;
        out_lse[row_base + r] = lse;
    }
}

} // namespace

// Forward: hidden [N, D] bf16, weight [V, D] bf16, target [N] int32, optional
// bias [V] f32. Returns (logp [N] f32, lse [N] f32). Logits are never
// materialized; peak extra memory is the per-CTA shared-memory tiles.
std::vector<torch::Tensor> fused_linear_logp_sm90_forward(torch::Tensor hidden,
                                                          torch::Tensor weight,
                                                          torch::Tensor target,
                                                          torch::optional<torch::Tensor> bias) {
    TORCH_CHECK(hidden.is_cuda() && weight.is_cuda(), "inputs must be CUDA tensors");
    TORCH_CHECK(hidden.scalar_type() == at::kBFloat16, "hidden must be bfloat16");
    TORCH_CHECK(weight.scalar_type() == at::kBFloat16, "weight must be bfloat16");
    TORCH_CHECK(hidden.is_contiguous() && weight.is_contiguous(), "inputs must be contiguous");
    const int N = hidden.size(0);
    const int D = hidden.size(1);
    const int V = weight.size(0);
    TORCH_CHECK(weight.size(1) == D, "hidden/weight hidden-dim mismatch");
    TORCH_CHECK(D % BK == 0, "D must be a multiple of ", BK, " for the SM90 kernel");

    auto opts_f = hidden.options().dtype(torch::kFloat);
    auto logp = torch::empty({N}, opts_f);
    auto lse = torch::empty({N}, opts_f);

    // TMA descriptors: box [rows=BM/BN, cols=BK]. cols*sizeof(bf16) = 32B; the
    // helper selects 32B swizzle for this stride -- the descriptor swizzle bits
    // in make_smem_desc must be kept consistent (see VALIDATION note above).
    CUtensorMap h_tmap, w_tmap;
    init_tensor_map(&h_tmap, reinterpret_cast<const nv_bfloat16 *>(hidden.data_ptr<at::BFloat16>()),
                    N, D, BM, BK);
    init_tensor_map(&w_tmap, reinterpret_cast<const nv_bfloat16 *>(weight.data_ptr<at::BFloat16>()),
                    V, D, BN, BK);

    const float *bias_ptr = nullptr;
    torch::Tensor bias_f;
    if (bias.has_value()) {
        bias_f = bias->to(torch::kFloat).contiguous();
        bias_ptr = bias_f.data_ptr<float>();
    }

    const int smem = (BM * BK + BN * BK) * sizeof(nv_bfloat16) + (BM * BN) * sizeof(float) +
                     3 * BM * sizeof(float) + 16;
    const int grid = (N + BM - 1) / BM;
    auto target_i = target.to(torch::kInt32).contiguous();

    fused_linear_logp_sm90_kernel<<<grid, WG_THREADS, smem>>>(
        h_tmap, w_tmap, target_i.data_ptr<int>(), bias_ptr, logp.data_ptr<float>(),
        lse.data_ptr<float>(), N, D, V);

    return {logp, lse};
}
