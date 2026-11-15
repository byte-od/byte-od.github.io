// ============================================================================
// PointScarFlow v1.0 — Pure Scar-Native 3D Point-Cloud Rectified Flow Shaders
//
// Executes 100% `ScarNativeLinear` (`D4-LUT256 VQ4-STE` + `FWHT-64` + `d_eq` +
// `INT4-STE` structural scars) batched 3D point-cloud synthesis on WebGPU WGSL 1.0.
//
// Strict Architectural Guarantees (`AGENTS.md`):
//   - Zero dense FP16/FP32 weight matrices (all linear projections decode directly
//     from `256 x vec4<f16>` `D4` codebooks with `LUT256[0] == (0,0,0,0)` + `INT4` scars)
//   - Zero `Math.random()` (deterministic canonical Fibonacci `S^2` Optimal-Transport flow)
//   - Baseline WebGPU compliant (`<= 12,288 B` workgroup shared memory, `<= 6` storage buffers)
// ============================================================================

struct PsfParams {
    p0: u32,
    p1: u32,
    p2: u32,
    p3: u32,
    p4: u32,
    p5: u32,
    p6: u32,
    p7: u32,
    f0: f32,
    f1: f32,
    f2: f32,
    f3: f32,
}

@group(0) @binding(0) var<uniform> params: PsfParams;
@group(0) @binding(1) var<storage, read> ro_u32_a: array<u32>;
@group(0) @binding(2) var<storage, read> ro_f32_a: array<f32>;
@group(0) @binding(3) var<storage, read> ro_f32_b: array<f32>;
@group(0) @binding(4) var<storage, read> ro_f32_c: array<f32>;
@group(0) @binding(5) var<storage, read_write> rw_f32_a: array<f32>;

fn psf_unpack_f16_lane(word: u32, elem_idx: u32) -> f32 {
    let pair = unpack2x16float(word);
    return select(pair.x, pair.y, (elem_idx & 1u) != 0u);
}

fn psf_rademacher_sign(idx: u32) -> f32 {
    var h: u32 = idx * 2654435761u;
    h = ((h >> 16u) ^ h) * 0x45D9F3Bu;
    return select(-1.0, 1.0, (h & 1u) == 0u);
}

// ----------------------------------------------------------------------------
// 1. Deterministic 64-Dim Geometric + Multi-Scale Fourier Point Stem Features
//    params: p0 = num_points, f0 = tau in [0, 1]
//    ro_f32_a: x_tau [N * 4] (`x, y, z, s`)
//    ro_f32_b: x_0   [N * 3] (canonical Fibonacci sphere anchor `x0, y0, z0`)
//    rw_f32_a: feat  [N * 64]
//    dispatch: ((num_points + 63u) >> 6u, 1u, 1u)
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn psf_build_point_features_f32(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let i = gid.x;
    if (i >= params.p0) {
        return;
    }
    let tau = params.f0;
    let b4 = i * 4u;
    let b3 = i * 3u;
    let out_base = i * 64u;

    let x = ro_f32_a[b4 + 0u];
    let y = ro_f32_a[b4 + 1u];
    let z = ro_f32_a[b4 + 2u];
    let s = ro_f32_a[b4 + 3u];

    let x0 = ro_f32_b[b3 + 0u];
    let y0 = ro_f32_b[b3 + 1u];
    let z0 = ro_f32_b[b3 + 2u];

    let dx = x - x0;
    let dy = y - y0;
    let dz = z - z0;
    let rad = sqrt(max(x * x + y * y + z * z, 1e-12));

    // [0..12]: raw state (4) + anchor (3) + displacement (3) + radial/time (3) = 13 floats
    rw_f32_a[out_base + 0u] = x;
    rw_f32_a[out_base + 1u] = y;
    rw_f32_a[out_base + 2u] = z;
    rw_f32_a[out_base + 3u] = s;
    rw_f32_a[out_base + 4u] = x0;
    rw_f32_a[out_base + 5u] = y0;
    rw_f32_a[out_base + 6u] = z0;
    rw_f32_a[out_base + 7u] = dx;
    rw_f32_a[out_base + 8u] = dy;
    rw_f32_a[out_base + 9u] = dz;
    rw_f32_a[out_base + 10u] = rad;
    rw_f32_a[out_base + 11u] = tau;
    rw_f32_a[out_base + 12u] = 1.0 - tau;

    // [13..36]: 4-octave Fourier features on (x, y, z) -> 4 * 6 = 24 floats
    let pi: f32 = 3.141592653589793;
    var idx: u32 = 13u;
    for (var f = 0u; f < 4u; f += 1u) {
        let omega = pi * f32(1u << f);
        rw_f32_a[out_base + idx + 0u] = sin(omega * x);
        rw_f32_a[out_base + idx + 1u] = cos(omega * x);
        rw_f32_a[out_base + idx + 2u] = sin(omega * y);
        rw_f32_a[out_base + idx + 3u] = cos(omega * y);
        rw_f32_a[out_base + idx + 4u] = sin(omega * z);
        rw_f32_a[out_base + idx + 5u] = cos(omega * z);
        idx += 6u;
    }

    // [37..60]: 4-octave Fourier features on canonical sphere anchor (x0, y0, z0) -> 24 floats
    for (var f = 0u; f < 4u; f += 1u) {
        let omega = pi * f32(1u << f);
        rw_f32_a[out_base + idx + 0u] = sin(omega * x0);
        rw_f32_a[out_base + idx + 1u] = cos(omega * x0);
        rw_f32_a[out_base + idx + 2u] = sin(omega * y0);
        rw_f32_a[out_base + idx + 3u] = cos(omega * y0);
        rw_f32_a[out_base + idx + 4u] = sin(omega * z0);
        rw_f32_a[out_base + idx + 5u] = cos(omega * z0);
        idx += 6u;
    }

    // [61..63]: Degree-2 spherical cross-harmonics (3 floats) -> exact 64 floats total!
    rw_f32_a[out_base + 61u] = x0 * y0;
    rw_f32_a[out_base + 62u] = y0 * z0;
    rw_f32_a[out_base + 63u] = z0 * x0;
}

// ----------------------------------------------------------------------------
// 2. Sinusoidal Flow-Time Embedding (`64` dims) + SiLU/Add Helper
//    params: f0 = tau in [0, 1]
//    rw_f32_a: t_emb [64]
//    dispatch: (1u, 1u, 1u)
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn psf_time_embed_64_f32(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let i = lid.x;
    let half = 32u;
    let tau = params.f0;
    let k = f32(i % half);
    // exp(-log(1000.0) * k / 31.0) * 64.0 * tau
    let freq = exp(-6.907755278982137 * (k / 31.0)) * 32.0;
    let ang = tau * freq;
    rw_f32_a[i] = select(cos(ang), sin(ang), i < half);
}

// ----------------------------------------------------------------------------
// 3. Batched Channel Equilibration + Orthonormal Block-64 FWHT-64
//    params: p0 = in_dim (multiple of 64), p1 = d_eq_u32_off, p2 = batch_size
//    ro_u32_a: shard containing FP16 `d_eq` [in_dim]
//    ro_f32_a: raw input `X_raw` [batch_size, in_dim]
//    rw_f32_a: rotated output `X_rot` [batch_size, in_dim]
//    dispatch: (in_dim >> 6u, batch_size, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_psf_fwht: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn psf_vq4_batched_equilibrate_fwht64(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let col = wgid.x * 64u + lane;
    let b = wgid.y;
    let in_dim = params.p0;
    let d_u32_off = params.p1;
    let idx = b * in_dim + col;

    let d_val = psf_unpack_f16_lane(ro_u32_a[d_u32_off + (col >> 1u)], col);
    wg_psf_fwht[lane] = (ro_f32_a[idx] / d_val) * psf_rademacher_sign(col);
    workgroupBarrier();

    for (var h = 1u; h < 64u; h <<= 1u) {
        let is_odd = (lane & h) != 0u;
        let partner = lane ^ h;
        let a = wg_psf_fwht[select(lane, partner, is_odd)];
        let bv = wg_psf_fwht[select(partner, lane, is_odd)];
        workgroupBarrier();
        wg_psf_fwht[lane] = select(a + bv, a - bv, is_odd);
        workgroupBarrier();
    }

    if (b < params.p2) {
        rw_f32_a[idx] = wg_psf_fwht[lane] * 0.125;
    }
}

// ----------------------------------------------------------------------------
// 4. Batched Zero-Barrier Block-64 INT4 `A_scar` Projection (`U = X_raw @ A_scar^T`)
//    1 thread per `(b, r)` element (`rank` is small: 8, 16, or 24; `0` barriers, `0 B` SRAM!).
//    params:
//      p0 = rank (multiple of 8)
//      p1 = in_dim (multiple of 64)
//      p2 = batch_size
//      p3 = ascar_packed_u32_off ([rank, in_dim / 8] u32 words)
//      p4 = ascar_scales_u32_off (FP16 [rank, in_dim / 64])
//    ro_u32_a: binary shard
//    ro_f32_a: `X_raw` [batch_size, in_dim]
//    rw_f32_a: `U` [batch_size, rank]
//    dispatch: ((batch_size * rank + 63u) >> 6u, 1u, 1u)
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn psf_batched_int4_ascar_gemm(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let idx = gid.x;
    let rank = params.p0;
    let total = params.p2 * rank;
    if (idx >= total) {
        return;
    }
    let b = idx / rank;
    let r = idx - b * rank;
    let in_dim = params.p1;
    let words_per_row = in_dim >> 3u;
    let blocks_per_row = in_dim >> 6u;
    let p_row_base = params.p3 + r * words_per_row;
    let s_row_base = r * blocks_per_row;
    let x_base = b * in_dim;

    var acc: f32 = 0.0;
    for (var blk = 0u; blk < blocks_per_row; blk += 1u) {
        let s_idx = s_row_base + blk;
        let scale = psf_unpack_f16_lane(ro_u32_a[params.p4 + (s_idx >> 1u)], s_idx);
        let w_base = p_row_base + (blk << 3u);
        let c_base = x_base + (blk << 6u);
        var blk_acc: f32 = 0.0;
        for (var w = 0u; w < 8u; w += 1u) {
            let word = ro_u32_a[w_base + w];
            let c0 = c_base + (w << 3u);
            blk_acc += (f32((word >>  0u) & 0xFu) - 8.0) * ro_f32_a[c0 + 0u]
                     + (f32((word >>  4u) & 0xFu) - 8.0) * ro_f32_a[c0 + 1u]
                     + (f32((word >>  8u) & 0xFu) - 8.0) * ro_f32_a[c0 + 2u]
                     + (f32((word >> 12u) & 0xFu) - 8.0) * ro_f32_a[c0 + 3u]
                     + (f32((word >> 16u) & 0xFu) - 8.0) * ro_f32_a[c0 + 4u]
                     + (f32((word >> 20u) & 0xFu) - 8.0) * ro_f32_a[c0 + 5u]
                     + (f32((word >> 24u) & 0xFu) - 8.0) * ro_f32_a[c0 + 6u]
                     + (f32((word >> 28u) & 0xFu) - 8.0) * ro_f32_a[c0 + 7u];
        }
        acc += blk_acc * scale;
    }
    rw_f32_a[idx] = acc;
}

// ----------------------------------------------------------------------------
// 5. Batched Register-Tiled `8 x 64` VQ4-LUT256 + INT4 `B_scar` GEMM (`6 KB` SRAM)
//    1 workgroup of 64 threads computes `[8 batch_items x 64 out_rows] = 512` outputs:
//      - Loads `wg_psf_lut` (`256 x vec4<f32>`, `4 KB`) once per workgroup
//      - Loads `[8 batch x 16 vec4<f32>]` (`2 KB`) of `X_rot` per 64-col block
//      - Each thread `tr` decodes `W_vq4[row]` ONCE in registers and reuses it
//        across all 8 batch items via hardware `dot(vec4<f32>, vec4<f32>)`!
//    dispatch: ((out_dim + 63u) >> 6u, (batch_size + 7u) >> 3u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_psf_lut: array<vec4<f32>, 256>;
var<workgroup> wg_psf_x4: array<vec4<f32>, 128>;

@compute @workgroup_size(64, 1, 1)
fn psf_batched_vq4_int4scar_gemm(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    for (var l = tid; l < 256u; l += 64u) {
        let lut_base = params.p5 + (l << 1u);
        let p01 = unpack2x16float(ro_u32_a[lut_base]);
        let p23 = unpack2x16float(ro_u32_a[lut_base + 1u]);
        wg_psf_lut[l] = vec4<f32>(p01.x, p01.y, p23.x, p23.y);
    }
    workgroupBarrier();

    let row = wgid.x * 64u + tid;
    let b_base = wgid.y * 8u;
    let out_dim = params.p0;
    let in_dim = params.p1;
    let rank = params.p2;
    let batch_size = params.p7;
    let num_blks = in_dim >> 6u;
    let vq_words_per_row = in_dim >> 4u;
    let b_cnt = min(8u, select(0u, batch_size - b_base, batch_size > b_base));

    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    var acc4: f32 = 0.0;
    var acc5: f32 = 0.0;
    var acc6: f32 = 0.0;
    var acc7: f32 = 0.0;

    for (var blk = 0u; blk < num_blks; blk += 1u) {
        // Load `[8 batch x 16 vec4<f32>]` = 128 vec4s (2 vec4s per thread)
        for (var idx = tid; idx < 128u; idx += 64u) {
            let tb = idx >> 4u;
            let g4 = idx & 15u;
            if (tb < b_cnt) {
                let x_src = (b_base + tb) * in_dim + blk * 64u + g4 * 4u;
                wg_psf_x4[idx] = vec4<f32>(
                    ro_f32_a[x_src + 0u],
                    ro_f32_a[x_src + 1u],
                    ro_f32_a[x_src + 2u],
                    ro_f32_a[x_src + 3u],
                );
            } else {
                wg_psf_x4[idx] = vec4<f32>(0.0);
            }
        }
        workgroupBarrier();

        if (row < out_dim) {
            let s_idx = row * num_blks + blk;
            let scale = psf_unpack_f16_lane(ro_u32_a[params.p4 + (s_idx >> 1u)], s_idx);
            let w_base = params.p3 + row * vq_words_per_row + blk * 4u;

            for (var w = 0u; w < 4u; w += 1u) {
                let word = ro_u32_a[w_base + w];
                let v0 = wg_psf_lut[(word >>  0u) & 0xFFu] * scale;
                let v1 = wg_psf_lut[(word >>  8u) & 0xFFu] * scale;
                let v2 = wg_psf_lut[(word >> 16u) & 0xFFu] * scale;
                let v3 = wg_psf_lut[(word >> 24u) & 0xFFu] * scale;
                let x_off = w * 4u;

                acc0 += dot(v0, wg_psf_x4[0u   + x_off]) + dot(v1, wg_psf_x4[0u   + x_off + 1u]) + dot(v2, wg_psf_x4[0u   + x_off + 2u]) + dot(v3, wg_psf_x4[0u   + x_off + 3u]);
                if (b_cnt > 1u) {
                    acc1 += dot(v0, wg_psf_x4[16u  + x_off]) + dot(v1, wg_psf_x4[16u  + x_off + 1u]) + dot(v2, wg_psf_x4[16u  + x_off + 2u]) + dot(v3, wg_psf_x4[16u  + x_off + 3u]);
                    acc2 += dot(v0, wg_psf_x4[32u  + x_off]) + dot(v1, wg_psf_x4[32u  + x_off + 1u]) + dot(v2, wg_psf_x4[32u  + x_off + 2u]) + dot(v3, wg_psf_x4[32u  + x_off + 3u]);
                    acc3 += dot(v0, wg_psf_x4[48u  + x_off]) + dot(v1, wg_psf_x4[48u  + x_off + 1u]) + dot(v2, wg_psf_x4[48u  + x_off + 2u]) + dot(v3, wg_psf_x4[48u  + x_off + 3u]);
                    acc4 += dot(v0, wg_psf_x4[64u  + x_off]) + dot(v1, wg_psf_x4[64u  + x_off + 1u]) + dot(v2, wg_psf_x4[64u  + x_off + 2u]) + dot(v3, wg_psf_x4[64u  + x_off + 3u]);
                    acc5 += dot(v0, wg_psf_x4[80u  + x_off]) + dot(v1, wg_psf_x4[80u  + x_off + 1u]) + dot(v2, wg_psf_x4[80u  + x_off + 2u]) + dot(v3, wg_psf_x4[80u  + x_off + 3u]);
                    acc6 += dot(v0, wg_psf_x4[96u  + x_off]) + dot(v1, wg_psf_x4[96u  + x_off + 1u]) + dot(v2, wg_psf_x4[96u  + x_off + 2u]) + dot(v3, wg_psf_x4[96u  + x_off + 3u]);
                    acc7 += dot(v0, wg_psf_x4[112u + x_off]) + dot(v1, wg_psf_x4[112u + x_off + 1u]) + dot(v2, wg_psf_x4[112u + x_off + 2u]) + dot(v3, wg_psf_x4[112u + x_off + 3u]);
                }
            }
        }
        workgroupBarrier();
    }

    if (row < out_dim && b_cnt > 0u) {
        var accs = array<f32, 8>(acc0, acc1, acc2, acc3, acc4, acc5, acc6, acc7);
        if (rank > 0u) {
            let b_words_per_row = rank >> 3u;
            let b_row_base = params.p6 + row * b_words_per_row;
            let b_scales_off = params.p6 + out_dim * b_words_per_row;
            let b_scale = psf_unpack_f16_lane(ro_u32_a[b_scales_off + (row >> 1u)], row);
            for (var bw = 0u; bw < b_words_per_row; bw += 1u) {
                let bword = ro_u32_a[b_row_base + bw];
                let bv0 = vec4<f32>(
                    f32((bword >>  0u) & 0xFu) - 8.0,
                    f32((bword >>  4u) & 0xFu) - 8.0,
                    f32((bword >>  8u) & 0xFu) - 8.0,
                    f32((bword >> 12u) & 0xFu) - 8.0,
                ) * b_scale;
                let bv1 = vec4<f32>(
                    f32((bword >> 16u) & 0xFu) - 8.0,
                    f32((bword >> 20u) & 0xFu) - 8.0,
                    f32((bword >> 24u) & 0xFu) - 8.0,
                    f32((bword >> 28u) & 0xFu) - 8.0,
                ) * b_scale;
                for (var tb = 0u; tb < b_cnt; tb += 1u) {
                    let r0 = (b_base + tb) * rank + (bw << 3u);
                    let u0 = vec4<f32>(ro_f32_b[r0 + 0u], ro_f32_b[r0 + 1u], ro_f32_b[r0 + 2u], ro_f32_b[r0 + 3u]);
                    let u1 = vec4<f32>(ro_f32_b[r0 + 4u], ro_f32_b[r0 + 5u], ro_f32_b[r0 + 6u], ro_f32_b[r0 + 7u]);
                    accs[tb] += dot(bv0, u0) + dot(bv1, u1);
                }
            }
        }
        for (var tb = 0u; tb < b_cnt; tb += 1u) {
            rw_f32_a[(b_base + tb) * out_dim + row] = accs[tb];
        }
    }
}

// ----------------------------------------------------------------------------
// 6. Batched RMSNorm + Optional AdaLN Modulation
//    params:
//      p0 = dim
//      p1 = norm_w_u32_off (FP16 [dim])
//      p2 = batch_size
//      p3 = use_adaln (0u or 1u)
//      p4 = shift_slot (index in `mod_vec` [6 * dim]: shift = slot*dim, scale = (slot+1)*dim)
//      f0 = eps
//    ro_u32_a: binary shard
//    ro_f32_a: input `X` [batch_size, dim]
//    ro_f32_b: `mod_vec` [6 * dim] (if `use_adaln == 1u`)
//    rw_f32_a: output `Y` [batch_size, dim]
//    dispatch: (batch_size, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_psf_norm: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn psf_rmsnorm_adaln_f32(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let b = wgid.x;
    let dim = params.p0;
    if (b >= params.p2) {
        return;
    }
    let row_off = b * dim;
    var sq_sum: f32 = 0.0;
    for (var c = tid; c < dim; c += 64u) {
        let v = ro_f32_a[row_off + c];
        sq_sum += v * v;
    }
    wg_psf_norm[tid] = sq_sum;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_psf_norm[tid] += wg_psf_norm[tid + s];
        }
        workgroupBarrier();
    }

    let rms_inv = inverseSqrt(wg_psf_norm[0] / f32(dim) + params.f0);
    let use_adaln = params.p3 != 0u;
    let shift_off = params.p4 * dim;
    let scale_off = (params.p4 + 1u) * dim;

    for (var c = tid; c < dim; c += 64u) {
        let w = psf_unpack_f16_lane(ro_u32_a[params.p1 + (c >> 1u)], c);
        var val = ro_f32_a[row_off + c] * rms_inv * w;
        if (use_adaln) {
            val = val * (1.0 + ro_f32_b[scale_off + c]) + ro_f32_b[shift_off + c];
        }
        rw_f32_a[row_off + c] = val;
    }
}

// ----------------------------------------------------------------------------
// 7A. Fast Zero-Loop-Barrier `Part <- Point` Cross-Attention (`head_dim == 32`)
//     `n_q = num_parts` (e.g. 16), `n_kv = num_points` (e.g. 1024).
//     Each workgroup of 64 threads handles ONE `(q_idx, head)` pair:
//       - Loads `q[0..31]` into `wg_p2p_q` (1 barrier)
//       - Each thread scans `kv = tid, tid + 64, ...` purely in registers (`0` barriers!)
//         computing exact online `(m_i, l_i, acc_i[0..31])`
//       - Reduces `(m, l, acc[0..31])` across the 64 threads in 6 log2 steps!
//     Shared memory: `(32 + 64 + 64 + 64 * 32) * 4 = 8,832 B` (`8.6 KB <= 16 KB`).
//     dispatch: (n_q, num_heads, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_p2p_q: array<f32, 32>;
var<workgroup> wg_p2p_m: array<f32, 64>;
var<workgroup> wg_p2p_l: array<f32, 64>;
var<workgroup> wg_p2p_acc: array<f32, 2048>;

@compute @workgroup_size(64, 1, 1)
fn psf_part_from_point_attn_hd32(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let q_idx = wgid.x;
    let h = wgid.y;
    let n_q = params.p0;
    let n_kv = params.p1;
    let num_heads = params.p2;
    let d_model = num_heads * 32u;
    let q_base = q_idx * d_model + h * 32u;

    if (tid < 32u) {
        wg_p2p_q[tid] = select(0.0, ro_f32_a[q_base + tid] * params.f0, q_idx < n_q);
    }
    workgroupBarrier();

    var m_loc: f32 = -1e30;
    var l_loc: f32 = 0.0;
    var acc_loc: array<f32, 32>;
    for (var d = 0u; d < 32u; d += 1u) {
        acc_loc[d] = 0.0;
    }

    for (var kv = tid; kv < n_kv; kv += 64u) {
        let kv_base = kv * d_model + h * 32u;
        var score: f32 = 0.0;
        for (var d = 0u; d < 32u; d += 1u) {
            score += wg_p2p_q[d] * ro_f32_b[kv_base + d];
        }
        let m_new = max(m_loc, score);
        let alpha = exp(clamp(m_loc - m_new, -80.0, 0.0));
        let beta = exp(clamp(score - m_new, -80.0, 0.0));
        l_loc = l_loc * alpha + beta;
        for (var d = 0u; d < 32u; d += 1u) {
            acc_loc[d] = acc_loc[d] * alpha + beta * ro_f32_c[kv_base + d];
        }
        m_loc = m_new;
    }

    wg_p2p_m[tid] = m_loc;
    wg_p2p_l[tid] = l_loc;
    let acc_base = tid * 32u;
    for (var d = 0u; d < 32u; d += 1u) {
        wg_p2p_acc[acc_base + d] = acc_loc[d];
    }
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            let m_a = wg_p2p_m[tid];
            let l_a = wg_p2p_l[tid];
            let m_b = wg_p2p_m[tid + s];
            let l_b = wg_p2p_l[tid + s];
            let m_ab = max(m_a, m_b);
            let wa = select(0.0, exp(clamp(m_a - m_ab, -80.0, 0.0)), l_a > 0.0);
            let wb = select(0.0, exp(clamp(m_b - m_ab, -80.0, 0.0)), l_b > 0.0);
            wg_p2p_m[tid] = m_ab;
            wg_p2p_l[tid] = l_a * wa + l_b * wb;
            let ba = tid * 32u;
            let bb = (tid + s) * 32u;
            for (var d = 0u; d < 32u; d += 1u) {
                wg_p2p_acc[ba + d] = wg_p2p_acc[ba + d] * wa + wg_p2p_acc[bb + d] * wb;
            }
        }
        workgroupBarrier();
    }

    if (tid < 32u && q_idx < n_q) {
        rw_f32_a[q_base + tid] = wg_p2p_acc[tid] / max(wg_p2p_l[0], 1e-12);
    }
}

// ----------------------------------------------------------------------------
// 7B. Fast SRAM-Cached `Point <- Part` Cross-Attention (`head_dim == 32`, `n_kv <= 32`)
//     Caches all `n_kv x 32` Key and Value vectors for `head = wgid.y` in `4 KB` SRAM
//     (`1` barrier total!), then 64 point-threads compute exact 32-dim attention in registers!
//     dispatch: ((n_q + 63u) >> 6u, num_heads, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_pt_k: array<f32, 1024>;
var<workgroup> wg_pt_v: array<f32, 1024>;

@compute @workgroup_size(64, 1, 1)
fn psf_point_from_part_attn_hd32(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let q_idx = wgid.x * 64u + tid;
    let h = wgid.y;
    let n_q = params.p0;
    let n_kv = params.p1;
    let num_heads = params.p2;
    let d_model = num_heads * 32u;
    let total_kv_elems = n_kv * 32u;

    for (var idx = tid; idx < total_kv_elems; idx += 64u) {
        let kv = idx >> 5u;
        let d = idx & 31u;
        let src = kv * d_model + h * 32u + d;
        wg_pt_k[idx] = ro_f32_b[src];
        wg_pt_v[idx] = ro_f32_c[src];
    }
    workgroupBarrier();

    if (q_idx >= n_q) {
        return;
    }

    let q_base = q_idx * d_model + h * 32u;
    var q_reg: array<f32, 32>;
    var acc_reg: array<f32, 32>;
    for (var d = 0u; d < 32u; d += 1u) {
        q_reg[d] = ro_f32_a[q_base + d] * params.f0;
        acc_reg[d] = 0.0;
    }

    var m_val: f32 = -1e30;
    var l_val: f32 = 0.0;
    for (var kv = 0u; kv < n_kv; kv += 1u) {
        let kv_off = kv * 32u;
        var score: f32 = 0.0;
        for (var d = 0u; d < 32u; d += 1u) {
            score += q_reg[d] * wg_pt_k[kv_off + d];
        }
        let m_new = max(m_val, score);
        let alpha = exp(clamp(m_val - m_new, -80.0, 0.0));
        let beta = exp(clamp(score - m_new, -80.0, 0.0));
        l_val = l_val * alpha + beta;
        for (var d = 0u; d < 32u; d += 1u) {
            acc_reg[d] = acc_reg[d] * alpha + beta * wg_pt_v[kv_off + d];
        }
        m_val = m_new;
    }

    let inv_l = 1.0 / max(l_val, 1e-12);
    for (var d = 0u; d < 32u; d += 1u) {
        rw_f32_a[q_base + d] = acc_reg[d] * inv_l;
    }
}

// ----------------------------------------------------------------------------
// 8. Elementwise Ops: SiLU, Add+SiLU, SwiGLU, Gated Residual Add, & Euler CFG Step
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn psf_add_silu_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.p0) {
        return;
    }
    // params.p1 == 1u: add `ro_f32_a + ro_f32_b` then SiLU; params.p1 == 0u: SiLU(`ro_f32_a`) only
    let v = select(ro_f32_a[i], ro_f32_a[i] + ro_f32_b[i], params.p1 != 0u);
    rw_f32_a[i] = v / (1.0 + exp(-clamp(v, -40.0, 40.0)));
}

@compute @workgroup_size(256, 1, 1)
fn psf_swiglu_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    if (i >= params.p0) {
        return;
    }
    let g = ro_f32_a[i];
    let sig = 1.0 / (1.0 + exp(-clamp(g, -40.0, 40.0)));
    rw_f32_a[i] = (g * sig) * ro_f32_b[i];
}

@compute @workgroup_size(256, 1, 1)
fn psf_gated_residual_add_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.p0;
    let dim = params.p1;
    if (i >= total) {
        return;
    }
    var scale: f32 = 1.0;
    if (params.p2 != 0u) {
        let gate_off = params.p3 * dim + (i % dim);
        scale = 1.0 + ro_f32_b[gate_off];
    }
    rw_f32_a[i] = rw_f32_a[i] + scale * ro_f32_a[i];
}

@compute @workgroup_size(256, 1, 1)
fn psf_euler_step_cfg_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.x;
    let total = params.p0;
    if (i >= total) {
        return;
    }
    let dt = params.f0;
    let cfg_w = params.f1;
    let v_cond = ro_f32_a[i];
    var v_eff = v_cond;
    if (params.p1 != 0u) {
        let v_uncond = ro_f32_b[i];
        v_eff = v_uncond + cfg_w * (v_cond - v_uncond);
    }
    var next_val = rw_f32_a[i] + dt * v_eff;
    // On final step (`params.p2 == 1u`), clamp coordinates to [-1.25, 1.25] and salience `s` (`i & 3 == 3`) to [0, 1]
    if (params.p2 != 0u) {
        if ((i & 3u) == 3u) {
            next_val = clamp(next_val, 0.0, 1.0);
        } else {
            next_val = clamp(next_val, -1.25, 1.25);
        }
    }
    rw_f32_a[i] = next_val;
}
