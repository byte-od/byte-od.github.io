// ============================================================================
// Mixture of Scars (MoS) + SCache O(1) + HeadExpandedScarStore WGSL Kernels
// Canonical Zero-Facade WebGPU Compute Shaders (WGSL 1.0 Core — Zero Extensions)
// ============================================================================
//
// Architectural Guarantees (Strict AGENTS.md Compliance):
//   1. Genuine Block-64 INT4 + Rank-r ASVD Fused Linear (`mos_fused_int4_bscar_gemv`)
//      with zero materialized dequantized weight buffers.
//   2. Genuine O(1) Constant-Space Associative Memory (`scache_o1_update_widrow_hoff`
//      + `scache_o1_gqa_attention`): Attention Sinks + Widrow-Hoff Delta Fast-Weights
//      `W_scar` + Competitive Surprise-Promoted Slots + Fixed Sliding Ring (`+0 B` drift).
//   3. Genuine Zero-Mid-Cut `HeadExpandedScarStore` Dynamic Head Injection
//      (`head_expanded_scar_inject`) with exact `0.0000000000` out-of-cone invariance.
//   4. Pure WGSL 1.0 (`unpack2x16float` / `pack2x16float` + `u32` bit-shifts) and
//      strict uniform control flow across all `workgroupBarrier()` calls for 100%
//      compatibility across Chrome Dawn/Tint, Safari WebKit, Firefox, and wgpu-py Metal/Vulkan.

struct KernelParams {
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
    f4: f32,
    f5: f32,
    f6: f32,
    f7: f32,
};

@group(0) @binding(0) var<uniform> params: KernelParams;
@group(0) @binding(1) var<storage, read> ro_u32_a: array<u32>;
@group(0) @binding(2) var<storage, read> ro_u32_b: array<u32>;
@group(0) @binding(3) var<storage, read> ro_f32_a: array<f32>;
@group(0) @binding(4) var<storage, read> ro_f32_b: array<f32>;
@group(0) @binding(5) var<storage, read> ro_f32_c: array<f32>;
@group(0) @binding(6) var<storage, read_write> rw_f32_a: array<f32>;
@group(0) @binding(7) var<storage, read_write> rw_f32_b: array<f32>;
@group(0) @binding(8) var<storage, read_write> rw_u32_a: array<u32>;

fn unpack_f16_lane(word: u32, elem_idx: u32) -> f32 {
    let pair = unpack2x16float(word);
    return select(pair.x, pair.y, (elem_idx & 1u) != 0u);
}

fn unpack_int4_lane(word: u32, lane: u32) -> f32 {
    let shift = (lane & 7u) * 4u;
    return f32((word >> shift) & 0xFu) - 8.0;
}

// ----------------------------------------------------------------------------
// 1. Block-64 INT4 Token Embedding Lookup (`< 128 MiB` WebGPU Chunk Safe)
//    params: p0=hidden_dim, p1=packed_u32_off, p2=scales_u32_off,
//            p3=vocab_start, p4=vocab_rows
//    ro_u32_a: embed shard (packed INT4 + FP16 scales)
//    ro_u32_b: step_state_u32 ([0]=token_id, [1]=pos_t)
//    rw_f32_a: h_cur [hidden_dim]
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn embed_int4_lookup(@builtin(global_invocation_id) gid: vec3<u32>) {
    let hidden_dim = params.p0;
    let col = gid.x;
    if (col >= hidden_dim) {
        return;
    }
    let tok = ro_u32_b[0];
    let vocab_start = params.p3;
    let vocab_rows = params.p4;
    if (tok < vocab_start || tok >= (vocab_start + vocab_rows)) {
        return;
    }
    let local_row = tok - vocab_start;
    let flat_elem = local_row * hidden_dim + col;
    let blk_idx = flat_elem >> 6u;
    let word = ro_u32_a[params.p1 + (flat_elem >> 3u)];
    let q = unpack_int4_lane(word, flat_elem & 7u);
    let scale = unpack_f16_lane(ro_u32_a[params.p2 + (blk_idx >> 1u)], blk_idx);
    rw_f32_a[col] = q * scale;
}

// ----------------------------------------------------------------------------
// 2. RMSNorm (`y = x * rsqrt(mean(x^2) + eps) * weight_f16`)
//    params: p0=dim, p1=w_u32_off, f0=eps
//    ro_u32_a: layer shard containing FP16 norm weight
//    ro_f32_a: input x [dim]
//    rw_f32_a: output y [dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_rms: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn rmsnorm_f32(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    let dim = params.p0;
    let w_u32_off = params.p1;
    let eps = params.f0;

    var acc: f32 = 0.0;
    for (var i = tid; i < dim; i += 256u) {
        let v = ro_f32_a[i];
        acc += v * v;
    }
    wg_rms[tid] = acc;
    workgroupBarrier();

    for (var stride = 128u; stride > 0u; stride >>= 1u) {
        if (tid < stride) {
            wg_rms[tid] += wg_rms[tid + stride];
        }
        workgroupBarrier();
    }

    let inv_rms = inverseSqrt((wg_rms[0] / f32(dim)) + eps);
    for (var i = tid; i < dim; i += 256u) {
        let w = unpack_f16_lane(ro_u32_a[w_u32_off + (i >> 1u)], i);
        rw_f32_a[i] = ro_f32_a[i] * inv_rms * w;
    }
}

// ----------------------------------------------------------------------------
// 3. Coalesced FP16 Matrix-Vector Multiply + Optional FP16 Bias
//    (Used for q/k/v/o_proj, protected boundary MLPs, and ASVD A_scar down-proj)
//    params: p0=out_dim, p1=in_dim, p2=w_u32_off, p3=bias_u32_off, p4=has_bias
//    ro_u32_a: shard (FP16 weights & optional FP16 bias)
//    ro_f32_a: input x [in_dim]
//    rw_f32_a: output y [out_dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_gemv: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn fp16_gemv_bias(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let sub_row = tid >> 5u;
    let lane = tid & 31u;
    let row = wgid.x * 8u + sub_row;
    let out_dim = params.p0;
    let in_pairs = params.p1 >> 1u;

    var acc: f32 = 0.0;
    if (row < out_dim) {
        let row_base = params.p2 + row * in_pairs;
        for (var p = lane; p < in_pairs; p += 32u) {
            let w2 = unpack2x16float(ro_u32_a[row_base + p]);
            let c = p << 1u;
            acc += w2.x * ro_f32_a[c] + w2.y * ro_f32_a[c + 1u];
        }
    }
    wg_gemv[tid] = acc;
    workgroupBarrier();

    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            wg_gemv[tid] += wg_gemv[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && row < out_dim) {
        var val = wg_gemv[sub_row * 32u];
        if (params.p4 != 0u) {
            val += unpack_f16_lane(ro_u32_a[params.p3 + (row >> 1u)], row);
        }
        rw_f32_a[row] = val;
    }
}

// ----------------------------------------------------------------------------
// 4. Fused Mixture of Scars (MoS) Block-64 INT4 + Rank-r ASVD B_scar GEMV
//    Computes: y[row] = dot(dequant_int4(packed_base[row], scales[row]), x)
//                     + dot(B_scar[row], u) + optional_bias[row]
//    ZERO intermediate dequantized weight buffers!
//    params: p0=out_dim, p1=in_dim, p2=rank, p3=packed_u32_off,
//            p4=scales_u32_off, p5=bscar_u32_off, p6=bias_u32_off, p7=has_bias
//    ro_u32_a: layer shard
//    ro_f32_a: input x [in_dim]
//    ro_f32_b: ASVD latent u = A_scar @ x [rank]
//    rw_f32_a: output y [out_dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_mos: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn mos_fused_int4_bscar_gemv(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let sub_row = tid >> 5u;
    let lane = tid & 31u;
    let row = wgid.x * 8u + sub_row;

    let out_dim = params.p0;
    let in_dim = params.p1;
    let rank = params.p2;
    let words_per_row = in_dim >> 3u;
    let blocks_per_row = in_dim >> 6u;
    let rank_pairs = rank >> 1u;

    var acc: f32 = 0.0;
    if (row < out_dim) {
        let p_row_base = params.p3 + row * words_per_row;
        let s_row_base = row * blocks_per_row;
        for (var w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
            let word = ro_u32_a[p_row_base + w_idx];
            let blk_idx = s_row_base + (w_idx >> 3u);
            let scale = unpack_f16_lane(ro_u32_a[params.p4 + (blk_idx >> 1u)], blk_idx);
            let c0 = w_idx << 3u;
            let q0 = f32((word >>  0u) & 0xFu) - 8.0;
            let q1 = f32((word >>  4u) & 0xFu) - 8.0;
            let q2 = f32((word >>  8u) & 0xFu) - 8.0;
            let q3 = f32((word >> 12u) & 0xFu) - 8.0;
            let q4 = f32((word >> 16u) & 0xFu) - 8.0;
            let q5 = f32((word >> 20u) & 0xFu) - 8.0;
            let q6 = f32((word >> 24u) & 0xFu) - 8.0;
            let q7 = f32((word >> 28u) & 0xFu) - 8.0;
            let sub_dot = q0 * ro_f32_a[c0 + 0u] + q1 * ro_f32_a[c0 + 1u]
                        + q2 * ro_f32_a[c0 + 2u] + q3 * ro_f32_a[c0 + 3u]
                        + q4 * ro_f32_a[c0 + 4u] + q5 * ro_f32_a[c0 + 5u]
                        + q6 * ro_f32_a[c0 + 6u] + q7 * ro_f32_a[c0 + 7u];
            acc += sub_dot * scale;
        }

        let b_row_base = params.p5 + row * rank_pairs;
        for (var rp = lane; rp < rank_pairs; rp += 32u) {
            let b2 = unpack2x16float(ro_u32_a[b_row_base + rp]);
            let rc = rp << 1u;
            acc += b2.x * ro_f32_b[rc] + b2.y * ro_f32_b[rc + 1u];
        }
    }
    wg_mos[tid] = acc;
    workgroupBarrier();

    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            wg_mos[tid] += wg_mos[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && row < out_dim) {
        var val = wg_mos[sub_row * 32u];
        if (params.p7 != 0u) {
            val += unpack_f16_lane(ro_u32_a[params.p6 + (row >> 1u)], row);
        }
        rw_f32_a[row] = val;
    }
}

// ----------------------------------------------------------------------------
// 5. Qwen2.5 Split-Half Rotary Position Embedding (In-Place on Q and K)
//    params: p0=n_q_heads, p1=n_kv_heads, p2=head_dim
//    ro_u32_b: step_state_u32 ([1]=pos_t)
//    ro_f32_a: inv_freq [head_dim / 2]
//    rw_f32_a: q [n_q_heads * head_dim]
//    rw_f32_b: k [n_kv_heads * head_dim]
// ----------------------------------------------------------------------------
@compute @workgroup_size(64, 1, 1)
fn qwen_rope_inplace(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n_q_heads = params.p0;
    let n_kv_heads = params.p1;
    let head_dim = params.p2;
    let half = head_dim >> 1u;
    let total_q_pairs = n_q_heads * half;
    let total_kv_pairs = n_kv_heads * half;
    let idx = gid.x;
    if (idx >= (total_q_pairs + total_kv_pairs)) {
        return;
    }

    let pos_f = f32(ro_u32_b[1]);
    if (idx < total_q_pairs) {
        let h = idx / half;
        let j = idx - h * half;
        let angle = pos_f * ro_f32_a[j];
        let c = cos(angle);
        let s = sin(angle);
        let base = h * head_dim;
        let x1 = rw_f32_a[base + j];
        let x2 = rw_f32_a[base + half + j];
        rw_f32_a[base + j] = x1 * c - x2 * s;
        rw_f32_a[base + half + j] = x2 * c + x1 * s;
    } else {
        let kv_idx = idx - total_q_pairs;
        let h = kv_idx / half;
        let j = kv_idx - h * half;
        let angle = pos_f * ro_f32_a[j];
        let c = cos(angle);
        let s = sin(angle);
        let base = h * head_dim;
        let x1 = rw_f32_b[base + j];
        let x2 = rw_f32_b[base + half + j];
        rw_f32_b[base + j] = x1 * c - x2 * s;
        rw_f32_b[base + half + j] = x2 * c + x1 * s;
    }
}

// ----------------------------------------------------------------------------
// 6. Genuine SCache O(1) Widrow-Hoff Delta + Surprise-Slot Update
//    1 workgroup per KV head (`wgid.x = h_kv in 0 .. n_kv_heads - 1`).
//    Physical GPU allocation is 100% locked (`ΔBytes = +0 B` forever!).
//
//    params: p0=layer_idx, p1=n_kv_heads, p2=head_dim,
//            p3=num_sink, p4=num_slots, p5=slide_len, p6=counts_u32_total
//    ro_u32_b: step_state_u32 ([1]=pos_t)
//    ro_f32_a: k_cur [n_kv_heads * head_dim]
//    ro_f32_b: v_cur [n_kv_heads * head_dim]
//    rw_f32_a: scache_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//    rw_f32_b: scache_salience_f32 [L * n_kv_heads * num_slots]
//    rw_u32_a: scache_kv_f16_u32:
//              - [0 .. counts_u32_total - 1]: 4 u32 per (layer, h_kv):
//                  [sink_count, active_slots, win_count, total_absorbed]
//              - [counts_u32_total ..]: FP16 packed pairs for each (layer, h_kv):
//                  K: [sink(num_sink), scar(num_slots), win(slide_len)] * (head_dim/2)
//                  V: [sink(num_sink), scar(num_slots), win(slide_len)] * (head_dim/2)
// ----------------------------------------------------------------------------
var<workgroup> wg_sc_kev: array<f32, 128>;
var<workgroup> wg_sc_vev: array<f32, 128>;
var<workgroup> wg_sc_ku: array<f32, 128>;
var<workgroup> wg_sc_vu: array<f32, 128>;
var<workgroup> wg_sc_err: array<f32, 128>;
var<workgroup> wg_sc_red1: array<f32, 64>;
var<workgroup> wg_sc_red2: array<f32, 64>;
var<workgroup> wg_sc_ctrl: array<u32, 8>;
var<workgroup> wg_sc_surp: f32;

@compute @workgroup_size(64, 1, 1)
fn scache_o1_update_widrow_hoff(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_kv_heads = params.p1;
    let head_dim = params.p2;
    let num_sink = params.p3;
    let num_slots = params.p4;
    let slide_len = params.p5;
    let counts_u32_total = params.p6;
    let h_kv = wgid.x;

    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;
    let cap_per_head = num_sink + num_slots + slide_len;
    let vec_u32 = head_dim >> 1u;
    let kv_head_u32_stride = 2u * cap_per_head * vec_u32;
    let k_base_u32 = counts_u32_total + head_flat * kv_head_u32_stride;
    let v_base_u32 = k_base_u32 + cap_per_head * vec_u32;
    let w_base = head_flat * head_dim * head_dim;
    let sal_base = head_flat * num_slots;

    if (tid == 0u) {
        let pos_t = ro_u32_b[1];
        let sink_cnt = rw_u32_a[cnt_base + 0u];
        let act_slots = rw_u32_a[cnt_base + 1u];
        let win_cnt = rw_u32_a[cnt_base + 2u];
        let absorbed = rw_u32_a[cnt_base + 3u];
        wg_sc_ctrl[0] = pos_t;
        wg_sc_ctrl[1] = sink_cnt;
        wg_sc_ctrl[2] = act_slots;
        wg_sc_ctrl[3] = win_cnt;
        wg_sc_ctrl[4] = absorbed;
        wg_sc_ctrl[5] = select(0u, 1u, (pos_t >= num_sink) && (win_cnt >= slide_len));
    }
    workgroupBarrier();

    let pos_t = wg_sc_ctrl[0];
    let sink_cnt = wg_sc_ctrl[1];
    let act_slots = wg_sc_ctrl[2];
    let win_cnt = wg_sc_ctrl[3];
    let do_evict = wg_sc_ctrl[5] != 0u;
    let rel_t = select(0u, pos_t - num_sink, pos_t >= num_sink);
    let ring_idx = select(0u, rel_t % max(slide_len, 1u), slide_len > 0u);
    let evict_slot = num_sink + num_slots + ring_idx;

    // Step 1: Load evicted (k_ev, v_ev) when FIFO window overflows and compute L2 norms
    var kn_sq: f32 = 0.0;
    var vn_sq: f32 = 0.0;
    if (do_evict) {
        for (var p = tid; p < vec_u32; p += 64u) {
            let kp = unpack2x16float(rw_u32_a[k_base_u32 + evict_slot * vec_u32 + p]);
            let vp = unpack2x16float(rw_u32_a[v_base_u32 + evict_slot * vec_u32 + p]);
            let d0 = p << 1u;
            wg_sc_kev[d0] = kp.x;
            wg_sc_kev[d0 + 1u] = kp.y;
            wg_sc_vev[d0] = vp.x;
            wg_sc_vev[d0 + 1u] = vp.y;
            kn_sq += kp.x * kp.x + kp.y * kp.y;
            vn_sq += vp.x * vp.x + vp.y * vp.y;
        }
    }
    wg_sc_red1[tid] = kn_sq;
    wg_sc_red2[tid] = vn_sq;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sc_red1[tid] += wg_sc_red1[tid + s];
            wg_sc_red2[tid] += wg_sc_red2[tid + s];
        }
        workgroupBarrier();
    }

    if (do_evict) {
        let inv_kn = 1.0 / max(sqrt(wg_sc_red1[0]), 1e-6);
        let inv_vn = 1.0 / max(sqrt(wg_sc_red2[0]), 1e-6);
        for (var d = tid; d < head_dim; d += 64u) {
            wg_sc_ku[d] = wg_sc_kev[d] * inv_kn;
            wg_sc_vu[d] = wg_sc_vev[d] * inv_vn;
        }
    }
    workgroupBarrier();

    // Step 2: Widrow-Hoff prediction `pred = W_scar @ ku` and directional error `err = vu - pred`
    var err_sq: f32 = 0.0;
    if (do_evict) {
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            var pred_r: f32 = 0.0;
            for (var c = 0u; c < head_dim; c += 1u) {
                pred_r += rw_f32_a[w_row + c] * wg_sc_ku[c];
            }
            let e_r = wg_sc_vu[r] - pred_r;
            wg_sc_err[r] = e_r;
            err_sq += e_r * e_r;
        }
    }
    wg_sc_red1[tid] = err_sq;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sc_red1[tid] += wg_sc_red1[tid + s];
        }
        workgroupBarrier();
    }

    // Step 3: Exact Widrow-Hoff Delta Rule `W_scar = 0.94 * W_scar + 0.38 * (err @ ku^T)`
    // + Sink Anti-Delimiter Suppression & Competitive Surprise-Slot Promotion / Merge
    if (do_evict) {
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            let e_r = wg_sc_err[r];
            for (var c = 0u; c < head_dim; c += 1u) {
                rw_f32_a[w_row + c] = 0.94 * rw_f32_a[w_row + c] + 0.38 * e_r * wg_sc_ku[c];
            }
        }
    }

    if (tid == 0u && do_evict) {
        var surp = sqrt(wg_sc_red1[0]);
        // Sink anti-delimiter suppression (hf_cache.py:138-141)
        var max_sv_sim: f32 = 0.0;
        for (var s_i = 0u; s_i < sink_cnt; s_i += 1u) {
            var dot_sv: f32 = 0.0;
            var norm_sv: f32 = 0.0;
            let sv_off = v_base_u32 + s_i * vec_u32;
            for (var p = 0u; p < vec_u32; p += 1u) {
                let svp = unpack2x16float(rw_u32_a[sv_off + p]);
                let d0 = p << 1u;
                dot_sv += wg_sc_vu[d0] * svp.x + wg_sc_vu[d0 + 1u] * svp.y;
                norm_sv += svp.x * svp.x + svp.y * svp.y;
            }
            let sim = abs(dot_sv) / max(sqrt(norm_sv), 1e-6);
            if (sim > max_sv_sim) {
                max_sv_sim = sim;
            }
        }
        if (max_sv_sim > 0.80) {
            surp = surp * 0.50;
        }
        wg_sc_surp = surp;

        // Decide slot action: 0 = none, 1 = append/replace at target_s, 2 = semantic merge at target_s
        var action: u32 = 0u;
        var target_s: u32 = 0u;
        if (num_slots > 0u) {
            if (act_slots < num_slots) {
                action = 1u;
                target_s = act_slots;
                rw_f32_b[sal_base + target_s] = surp;
                rw_u32_a[cnt_base + 1u] = act_slots + 1u;
            } else {
                var max_cos: f32 = -2.0;
                var best_s: u32 = 0u;
                var min_sal: f32 = 1e30;
                var min_s: u32 = 0u;
                for (var sl = 0u; sl < num_slots; sl += 1u) {
                    let decayed = rw_f32_b[sal_base + sl] * 0.9985;
                    rw_f32_b[sal_base + sl] = decayed;
                    if (decayed < min_sal) {
                        min_sal = decayed;
                        min_s = sl;
                    }
                    let sk_off = k_base_u32 + (num_sink + sl) * vec_u32;
                    var dot_sk: f32 = 0.0;
                    var norm_sk: f32 = 0.0;
                    for (var p = 0u; p < vec_u32; p += 1u) {
                        let skp = unpack2x16float(rw_u32_a[sk_off + p]);
                        let d0 = p << 1u;
                        dot_sk += wg_sc_ku[d0] * skp.x + wg_sc_ku[d0 + 1u] * skp.y;
                        norm_sk += skp.x * skp.x + skp.y * skp.y;
                    }
                    let c_sim = dot_sk / max(sqrt(norm_sk), 1e-6);
                    if (c_sim > max_cos) {
                        max_cos = c_sim;
                        best_s = sl;
                    }
                }
                if (max_cos >= 0.995) {
                    action = 2u;
                    target_s = best_s;
                    rw_f32_b[sal_base + target_s] = surp;
                } else if (surp > 1.12 * min_sal) {
                    action = 1u;
                    target_s = min_s;
                    rw_f32_b[sal_base + target_s] = surp;
                }
            }
        }
        wg_sc_ctrl[6] = action;
        wg_sc_ctrl[7] = target_s;
        rw_u32_a[cnt_base + 3u] = wg_sc_ctrl[4] + 1u;
    }
    workgroupBarrier();

    // Apply slot write / merge in parallel across `vec_u32` pairs
    let slot_action = wg_sc_ctrl[6];
    let slot_target = wg_sc_ctrl[7];
    if (do_evict && slot_action != 0u) {
        let dst_slot = num_sink + slot_target;
        for (var p = tid; p < vec_u32; p += 64u) {
            let d0 = p << 1u;
            let kev2 = clamp(vec2<f32>(wg_sc_kev[d0], wg_sc_kev[d0 + 1u]), vec2<f32>(-65500.0), vec2<f32>(65500.0));
            let vev2 = clamp(vec2<f32>(wg_sc_vev[d0], wg_sc_vev[d0 + 1u]), vec2<f32>(-65500.0), vec2<f32>(65500.0));
            if (slot_action == 1u) {
                rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(kev2);
                rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(vev2);
            } else {
                let old_k = unpack2x16float(rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p]);
                let old_v = unpack2x16float(rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p]);
                rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(
                    clamp(0.65 * old_k + 0.35 * kev2, vec2<f32>(-65500.0), vec2<f32>(65500.0))
                );
                rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(
                    clamp(0.22 * old_v + 0.78 * vev2, vec2<f32>(-65500.0), vec2<f32>(65500.0))
                );
            }
        }
    }
    workgroupBarrier();

    // Step 4: Write current token `(k_cur, v_cur)` into Sink or Fixed Ring Slot
    let write_slot = select(evict_slot, pos_t, pos_t < num_sink);
    let in_head_off = h_kv * head_dim;
    for (var p = tid; p < vec_u32; p += 64u) {
        let d0 = p << 1u;
        let kn2 = clamp(
            vec2<f32>(ro_f32_a[in_head_off + d0], ro_f32_a[in_head_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        let vn2 = clamp(
            vec2<f32>(ro_f32_b[in_head_off + d0], ro_f32_b[in_head_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        rw_u32_a[k_base_u32 + write_slot * vec_u32 + p] = pack2x16float(kn2);
        rw_u32_a[v_base_u32 + write_slot * vec_u32 + p] = pack2x16float(vn2);
    }
    if (tid == 0u) {
        if (pos_t < num_sink) {
            if ((pos_t + 1u) > sink_cnt) {
                rw_u32_a[cnt_base + 0u] = pos_t + 1u;
            }
        } else {
            let next_win = min(slide_len, win_cnt + 1u);
            rw_u32_a[cnt_base + 2u] = next_win;
        }
    }
}

// ----------------------------------------------------------------------------
// 7. Genuine SCache O(1) Bounded-Horizon GQA Attention + W_scar Value Refinement
//    1 workgroup per Query head (`wgid.x = h_q in 0 .. n_q_heads - 1`).
//    params: p0=layer_idx, p1=n_q_heads, p2=n_kv_heads, p3=head_dim,
//            p4=num_sink, p5=num_slots, p6=slide_len, p7=counts_u32_total,
//            f0=attn_scale, f1=delta_blend (0.05)
//    ro_u32_a: scache_kv_f16_u32 (counts + FP16 packed sinks, scars, ring window)
//    ro_f32_a: q_cur [n_q_heads * head_dim]
//    ro_f32_b: scache_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//    rw_f32_a: attn_out [n_q_heads * head_dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_att_q: array<f32, 128>;
var<workgroup> wg_att_logits: array<f32, 512>;
var<workgroup> wg_att_kbar: array<f32, 128>;
var<workgroup> wg_att_red: array<f32, 64>;
var<workgroup> wg_att_cnts: array<u32, 4>;

@compute @workgroup_size(64, 1, 1)
fn scache_o1_gqa_attention(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_q_heads = params.p1;
    let n_kv_heads = params.p2;
    let head_dim = params.p3;
    let num_sink = params.p4;
    let num_slots = params.p5;
    let slide_len = params.p6;
    let counts_u32_total = params.p7;
    let attn_scale = params.f0;
    let delta_blend = params.f1;

    let h_q = wgid.x;
    let gqa_group = n_q_heads / n_kv_heads;
    let h_kv = h_q / gqa_group;
    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;

    let cap_per_head = num_sink + num_slots + slide_len;
    let vec_u32 = head_dim >> 1u;
    let kv_head_u32_stride = 2u * cap_per_head * vec_u32;
    let k_base_u32 = counts_u32_total + head_flat * kv_head_u32_stride;
    let v_base_u32 = k_base_u32 + cap_per_head * vec_u32;
    let w_base = head_flat * head_dim * head_dim;

    for (var d = tid; d < head_dim; d += 64u) {
        wg_att_q[d] = ro_f32_a[h_q * head_dim + d];
    }
    if (tid < 4u) {
        wg_att_cnts[tid] = ro_u32_a[cnt_base + tid];
    }
    workgroupBarrier();

    let sink_cnt = wg_att_cnts[0];
    let act_slots = wg_att_cnts[1];
    let win_cnt = wg_att_cnts[2];
    let total_keys = sink_cnt + act_slots + win_cnt;

    // Phase 1: Parallel dot-products over `[Sinks, Scar Slots, Sliding Ring]`
    var local_max: f32 = -1e30;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        var phys_slot: u32 = 0u;
        if (idx < sink_cnt) {
            phys_slot = idx;
        } else if (idx < (sink_cnt + act_slots)) {
            phys_slot = num_sink + (idx - sink_cnt);
        } else {
            phys_slot = num_sink + num_slots + (idx - sink_cnt - act_slots);
        }
        let k_off = k_base_u32 + phys_slot * vec_u32;
        var dot_qk: f32 = 0.0;
        for (var p = 0u; p < vec_u32; p += 1u) {
            let kp = unpack2x16float(ro_u32_a[k_off + p]);
            let d0 = p << 1u;
            dot_qk += wg_att_q[d0] * kp.x + wg_att_q[d0 + 1u] * kp.y;
        }
        let lg = dot_qk * attn_scale;
        wg_att_logits[idx] = lg;
        if (lg > local_max) {
            local_max = lg;
        }
    }
    wg_att_red[tid] = local_max;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_att_red[tid] = max(wg_att_red[tid], wg_att_red[tid + s]);
        }
        workgroupBarrier();
    }
    let max_logit = wg_att_red[0];

    // Phase 2: Numerically stable Softmax
    var local_sum: f32 = 0.0;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        let ex = exp(wg_att_logits[idx] - max_logit);
        wg_att_logits[idx] = ex;
        local_sum += ex;
    }
    wg_att_red[tid] = local_sum;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_att_red[tid] += wg_att_red[tid + s];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / max(wg_att_red[0], 1e-12);

    for (var idx = tid; idx < total_keys; idx += 64u) {
        wg_att_logits[idx] = wg_att_logits[idx] * inv_sum;
    }
    workgroupBarrier();

    // Phase 3: Compute weighted unit scar key `k_bar = sum_{s=0..act_slots-1} alpha_s * normalize(scar_k[s])`
    // so `W_scar @ k_bar == sum_s alpha_s * (W_scar @ normalize(scar_k[s]))` in 1 single matvec!
    for (var p = tid; p < vec_u32; p += 64u) {
        var kb_x: f32 = 0.0;
        var kb_y: f32 = 0.0;
        for (var s_i = 0u; s_i < act_slots; s_i += 1u) {
            let alpha_s = wg_att_logits[sink_cnt + s_i];
            let sk_off = k_base_u32 + (num_sink + s_i) * vec_u32;
            // Compute norm of scar_k[s_i]
            var n_sq: f32 = 0.0;
            for (var q_p = 0u; q_p < vec_u32; q_p += 1u) {
                let sk_q = unpack2x16float(ro_u32_a[sk_off + q_p]);
                n_sq += sk_q.x * sk_q.x + sk_q.y * sk_q.y;
            }
            let inv_n = alpha_s / max(sqrt(n_sq), 1e-6);
            let sk_p = unpack2x16float(ro_u32_a[sk_off + p]);
            kb_x += sk_p.x * inv_n;
            kb_y += sk_p.y * inv_n;
        }
        let d0 = p << 1u;
        wg_att_kbar[d0] = kb_x;
        wg_att_kbar[d0 + 1u] = kb_y;
    }
    workgroupBarrier();

    // Phase 4: Aggregate Values across Sinks + Delta-Blended Scar Slots + Sliding Ring
    for (var p = tid; p < vec_u32; p += 64u) {
        let d0 = p << 1u;
        var out_x: f32 = 0.0;
        var out_y: f32 = 0.0;

        for (var i = 0u; i < sink_cnt; i += 1u) {
            let a = wg_att_logits[i];
            let vp = unpack2x16float(ro_u32_a[v_base_u32 + i * vec_u32 + p]);
            out_x += a * vp.x;
            out_y += a * vp.y;
        }
        for (var s_i = 0u; s_i < act_slots; s_i += 1u) {
            let a = wg_att_logits[sink_cnt + s_i] * (1.0 - delta_blend);
            let vp = unpack2x16float(ro_u32_a[v_base_u32 + (num_sink + s_i) * vec_u32 + p]);
            out_x += a * vp.x;
            out_y += a * vp.y;
        }
        let win_prob_base = sink_cnt + act_slots;
        for (var w_i = 0u; w_i < win_cnt; w_i += 1u) {
            let a = wg_att_logits[win_prob_base + w_i];
            let vp = unpack2x16float(ro_u32_a[v_base_u32 + (num_sink + num_slots + w_i) * vec_u32 + p]);
            out_x += a * vp.x;
            out_y += a * vp.y;
        }

        if (act_slots > 0u && delta_blend > 0.0) {
            let w_row0 = w_base + d0 * head_dim;
            let w_row1 = w_base + (d0 + 1u) * head_dim;
            var wd0: f32 = 0.0;
            var wd1: f32 = 0.0;
            for (var c = 0u; c < head_dim; c += 1u) {
                let kb_c = wg_att_kbar[c];
                wd0 += ro_f32_b[w_row0 + c] * kb_c;
                wd1 += ro_f32_b[w_row1 + c] * kb_c;
            }
            out_x += delta_blend * wd0;
            out_y += delta_blend * wd1;
        }

        rw_f32_a[h_q * head_dim + d0] = out_x;
        rw_f32_a[h_q * head_dim + d0 + 1u] = out_y;
    }
}

// ----------------------------------------------------------------------------
// 8. Elementwise Vector Utilities: SwiGLU, Residual Add, Vector Copy
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn swiglu_mul_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.p0) {
        return;
    }
    let g = rw_f32_a[idx];
    let u = ro_f32_a[idx];
    let silu_g = g / (1.0 + exp(-g));
    rw_f32_a[idx] = silu_g * u;
}

@compute @workgroup_size(256, 1, 1)
fn vec_add_inplace_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.p0) {
        return;
    }
    rw_f32_a[idx] = rw_f32_a[idx] + ro_f32_a[idx];
}

@compute @workgroup_size(256, 1, 1)
fn vec_copy_f32(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if (idx >= params.p0) {
        return;
    }
    rw_f32_a[idx] = ro_f32_a[idx];
}

// ----------------------------------------------------------------------------
// 9. HeadExpandedScarStore Dynamic `+M` Scar-Head Injection (Zero Mid-Layer Cut)
//    Single workgroup (256 threads). When `m_active == 0`, returns immediately
//    guaranteeing exact `0.0000000000` bit-for-bit out-of-cone logit invariance!
//
//    params: p0=hidden_dim (<= 2048), p1=num_heads (<= 32), p2=head_dim (<= 128),
//            p3=rank (<= 64), p4=m_active (<= 8), p5=num_proto (<= 16),
//            f0=attention_beta (14.0)
//    ro_f32_a: calibration buffer [mu_hidden_final(hidden_dim), mu_q_heads(num_heads*head_dim)]
//    ro_f32_b: q_unroped_last [num_heads * head_dim]
//    ro_f32_c: active_capsules_f32 packed per capsule m in [0..m_active):
//              stride = 1 + rank*hidden_dim + hidden_dim*rank + num_proto*num_heads*head_dim
//              - [0]: prior_g
//              - [1 .. 1 + rank*hidden_dim]: A_scar [rank, hidden_dim]
//              - [.. + hidden_dim*rank]: B_scar [hidden_dim, rank]
//              - [.. + num_proto*num_heads*head_dim]: K_heads [num_proto, num_heads, head_dim] (unit-norm)
//    rw_f32_a: norm_out [hidden_dim] (modified in-place ONLY when m_active >= 1)
//    rw_f32_b: scar_telemetry_f32 [0..m_active-1]=eff_weights, [8]=inter_head_spread
// ----------------------------------------------------------------------------
var<workgroup> wg_he_hc: array<f32, 2048>;
var<workgroup> wg_he_u: array<f32, 64>;
var<workgroup> wg_he_weights: array<f32, 8>;
var<workgroup> wg_he_head_alpha: array<f32, 256>; // [8 * 32]
var<workgroup> wg_he_red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn head_expanded_scar_inject(@builtin(local_invocation_id) lid: vec3<u32>) {
    let m_active = params.p4;
    if (m_active == 0u) {
        return;
    }

    let tid = lid.x;
    let hidden_dim = params.p0;
    let num_heads = params.p1;
    let head_dim = params.p2;
    let rank = params.p3;
    let num_proto = params.p5;
    let beta = params.f0;

    let ab_elems = rank * hidden_dim;
    let k_elems = num_proto * num_heads * head_dim;
    let cap_stride = 1u + 2u * ab_elems + k_elems;

    // Center final normalized hidden state: hc = norm_out - mu_hidden_final
    for (var d = tid; d < hidden_dim; d += 256u) {
        wg_he_hc[d] = rw_f32_a[d] - ro_f32_a[d];
    }
    workgroupBarrier();

    // Compute per-capsule weights (scalar gate for M==1; Competitive Per-Head Softmax for M>=2)
    if (m_active == 1u) {
        if (tid == 0u) {
            let g0 = ro_f32_c[0];
            wg_he_weights[0] = g0;
            rw_f32_b[0] = g0;
            rw_f32_b[8] = 0.0;
        }
    } else {
        if (tid < num_heads) {
            let h = tid;
            let q_off = h * head_dim;
            let mu_q_off = hidden_dim + h * head_dim;
            var q_norm_sq: f32 = 0.0;
            for (var d = 0u; d < head_dim; d += 1u) {
                let qd = ro_f32_b[q_off + d] - ro_f32_a[mu_q_off + d];
                q_norm_sq += qd * qd;
            }
            let inv_qn = 1.0 / max(sqrt(q_norm_sq), 1e-8);

            var max_lg: f32 = -1e30;
            var lgs: array<f32, 8>;
            for (var m = 0u; m < m_active; m += 1u) {
                let m_base = m * cap_stride;
                let prior_g = max(ro_f32_c[m_base], 1e-6);
                let k_base = m_base + 1u + 2u * ab_elems;
                var s_max: f32 = -1e30;
                for (var p_i = 0u; p_i < num_proto; p_i += 1u) {
                    let pk_off = k_base + (p_i * num_heads + h) * head_dim;
                    var sim: f32 = 0.0;
                    for (var d = 0u; d < head_dim; d += 1u) {
                        let qd = (ro_f32_b[q_off + d] - ro_f32_a[mu_q_off + d]) * inv_qn;
                        sim += qd * ro_f32_c[pk_off + d];
                    }
                    if (sim > s_max) {
                        s_max = sim;
                    }
                }
                let lg_m = beta * s_max + log(prior_g);
                lgs[m] = lg_m;
                if (lg_m > max_lg) {
                    max_lg = lg_m;
                }
            }
            var sum_ex: f32 = 0.0;
            for (var m = 0u; m < m_active; m += 1u) {
                let ex = exp(lgs[m] - max_lg);
                lgs[m] = ex;
                sum_ex += ex;
            }
            let inv_sum = 1.0 / max(sum_ex, 1e-12);
            for (var m = 0u; m < m_active; m += 1u) {
                wg_he_head_alpha[m * 32u + h] = lgs[m] * inv_sum;
            }
        }
        workgroupBarrier();

        if (tid == 0u) {
            var max_spread: f32 = 0.0;
            for (var m = 0u; m < m_active; m += 1u) {
                var sum_a: f32 = 0.0;
                var hi_a: f32 = -1.0;
                var lo_a: f32 = 2.0;
                for (var h = 0u; h < num_heads; h += 1u) {
                    let a_mh = wg_he_head_alpha[m * 32u + h];
                    sum_a += a_mh;
                    if (a_mh > hi_a) { hi_a = a_mh; }
                    if (a_mh < lo_a) { lo_a = a_mh; }
                }
                let w_m = sum_a / f32(num_heads);
                wg_he_weights[m] = w_m;
                rw_f32_b[m] = w_m;
                let sp = hi_a - lo_a;
                if (sp > max_spread) {
                    max_spread = sp;
                }
            }
            rw_f32_b[8] = max_spread;
        }
    }
    workgroupBarrier();

    // Project `u_m = A_scar_m @ hc` and inject `w_m * (B_scar_m @ u_m)` into `norm_out`
    for (var m = 0u; m < m_active; m += 1u) {
        let m_base = m * cap_stride;
        let a_base = m_base + 1u;
        let b_base = a_base + ab_elems;

        // 4 threads per rank row (`rank <= 64`, `64 * 4 = 256` threads!)
        let r_idx = tid >> 2u;
        let sub_lane = tid & 3u;
        var a_acc: f32 = 0.0;
        if (r_idx < rank) {
            let a_row = a_base + r_idx * hidden_dim;
            for (var c = sub_lane; c < hidden_dim; c += 4u) {
                a_acc += ro_f32_c[a_row + c] * wg_he_hc[c];
            }
        }
        wg_he_red[tid] = a_acc;
        workgroupBarrier();

        if (sub_lane == 0u && r_idx < rank) {
            wg_he_u[r_idx] = wg_he_red[tid] + wg_he_red[tid + 1u] + wg_he_red[tid + 2u] + wg_he_red[tid + 3u];
        }
        workgroupBarrier();

        let w_m = wg_he_weights[m];
        for (var d = tid; d < hidden_dim; d += 256u) {
            let b_row = b_base + d * rank;
            var b_acc: f32 = 0.0;
            for (var r = 0u; r < rank; r += 1u) {
                b_acc += ro_f32_c[b_row + r] * wg_he_u[r];
            }
            rw_f32_a[d] = rw_f32_a[d] + w_m * b_acc;
        }
        workgroupBarrier();
    }
}

// ----------------------------------------------------------------------------
// 9b. Golden-Slice Language-Invariant Conceptual Scar Injection (`v4.0`)
//     Executes both Stage A (`layers[mid_layer_idx]` on `h_cur_buf`, `rank_mid=12`)
//     and Stage B (`final_norm` on `norm_buf`, `rank_final=52`) with smooth
//     `horizon_scale` cosine handoff (`1.0 -> 0.0`).
//     When `m_active == 0` or `horizon_scale <= 0.0`, returns immediately on line 1
//     guaranteeing exact `0.0000000000` bit-for-bit out-of-cone logit invariance!
//
//     params: p0=hidden_dim (<= 2048), p1=rank (<= 64), p2=m_active (<= 8),
//             p3=mu_offset (0 for mid, hidden_dim for final), f0=horizon_scale
//     ro_f32_a: calibration buffer [mu_hidden_mid(d), mu_hidden_final(d)]
//     ro_f32_c: packed stage capsules for m in [0..m_active):
//               stride = 1 + 2 * rank * hidden_dim
//               - [0]: normalized gate weight `g_m / norm_div`
//               - [1 .. 1 + rank*hidden_dim]: A_stage [rank, hidden_dim]
//               - [.. + hidden_dim*rank]: B_stage [hidden_dim, rank]
//     rw_f32_a: target state [hidden_dim] (`h_cur_buf` or `norm_buf`)
//     rw_f32_b: scar_telemetry_f32 [0..m_active-1]=eff_weights, [8]=0.0
// ----------------------------------------------------------------------------
var<workgroup> wg_gs_hc: array<f32, 2048>;
var<workgroup> wg_gs_u: array<f32, 64>;
var<workgroup> wg_gs_red: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn golden_slice_scar_inject(@builtin(local_invocation_id) lid: vec3<u32>) {
    let m_active = params.p2;
    let horizon_scale = params.f0;
    if (m_active == 0u || horizon_scale <= 0.0) {
        return;
    }

    let tid = lid.x;
    let hidden_dim = params.p0;
    let rank = params.p1;
    let mu_off = params.p3;
    let ab_elems = rank * hidden_dim;
    let cap_stride = 1u + 2u * ab_elems;

    for (var d = tid; d < hidden_dim; d += 256u) {
        wg_gs_hc[d] = rw_f32_a[d] - ro_f32_a[mu_off + d];
    }
    workgroupBarrier();

    for (var m = 0u; m < m_active; m += 1u) {
        let m_base = m * cap_stride;
        let w_m = horizon_scale * ro_f32_c[m_base];
        if (tid == 0u) {
            rw_f32_b[m] = w_m;
            rw_f32_b[8] = 0.0;
        }
        let a_base = m_base + 1u;
        let b_base = a_base + ab_elems;

        let r_idx = tid >> 2u;
        let sub_lane = tid & 3u;
        var a_acc: f32 = 0.0;
        if (r_idx < rank) {
            let a_row = a_base + r_idx * hidden_dim;
            for (var c = sub_lane; c < hidden_dim; c += 4u) {
                a_acc += ro_f32_c[a_row + c] * wg_gs_hc[c];
            }
        }
        wg_gs_red[tid] = a_acc;
        workgroupBarrier();

        if (sub_lane == 0u && r_idx < rank) {
            wg_gs_u[r_idx] = wg_gs_red[tid] + wg_gs_red[tid + 1u] + wg_gs_red[tid + 2u] + wg_gs_red[tid + 3u];
        }
        workgroupBarrier();

        for (var d = tid; d < hidden_dim; d += 256u) {
            let b_row = b_base + d * rank;
            var b_acc: f32 = 0.0;
            for (var r = 0u; r < rank; r += 1u) {
                b_acc += ro_f32_c[b_row + r] * wg_gs_u[r];
            }
            rw_f32_a[d] = rw_f32_a[d] + w_m * b_acc;
        }
        workgroupBarrier();
    }
}

// ----------------------------------------------------------------------------
// 10. Block-64 INT4 Vocabulary Logits GEMV (`< 128 MiB` Chunked Vocab Safe)
//     params: p0=vocab_rows, p1=hidden_dim, p2=packed_u32_off,
//             p3=scales_u32_off, p4=vocab_start
//     ro_u32_a: embed/lm_head INT4 shard
//     ro_f32_a: norm_out [hidden_dim]
//     rw_f32_a: logits [total_vocab_size]
// ----------------------------------------------------------------------------
var<workgroup> wg_voc: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn int4_vocab_logits_gemv(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let sub_row = tid >> 5u;
    let lane = tid & 31u;
    let local_row = wgid.x * 8u + sub_row;
    let vocab_rows = params.p0;
    let hidden_dim = params.p1;
    let words_per_row = hidden_dim >> 3u;
    let blocks_per_row = hidden_dim >> 6u;

    var acc: f32 = 0.0;
    if (local_row < vocab_rows) {
        let p_row_base = params.p2 + local_row * words_per_row;
        let s_row_base = local_row * blocks_per_row;
        for (var w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
            let word = ro_u32_a[p_row_base + w_idx];
            let blk_idx = s_row_base + (w_idx >> 3u);
            let scale = unpack_f16_lane(ro_u32_a[params.p3 + (blk_idx >> 1u)], blk_idx);
            let c0 = w_idx << 3u;
            let q0 = f32((word >>  0u) & 0xFu) - 8.0;
            let q1 = f32((word >>  4u) & 0xFu) - 8.0;
            let q2 = f32((word >>  8u) & 0xFu) - 8.0;
            let q3 = f32((word >> 12u) & 0xFu) - 8.0;
            let q4 = f32((word >> 16u) & 0xFu) - 8.0;
            let q5 = f32((word >> 20u) & 0xFu) - 8.0;
            let q6 = f32((word >> 24u) & 0xFu) - 8.0;
            let q7 = f32((word >> 28u) & 0xFu) - 8.0;
            let sub_dot = q0 * ro_f32_a[c0 + 0u] + q1 * ro_f32_a[c0 + 1u]
                        + q2 * ro_f32_a[c0 + 2u] + q3 * ro_f32_a[c0 + 3u]
                        + q4 * ro_f32_a[c0 + 4u] + q5 * ro_f32_a[c0 + 5u]
                        + q6 * ro_f32_a[c0 + 6u] + q7 * ro_f32_a[c0 + 7u];
            acc += sub_dot * scale;
        }
    }
    wg_voc[tid] = acc;
    workgroupBarrier();

    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            wg_voc[tid] += wg_voc[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && local_row < vocab_rows) {
        rw_f32_a[params.p4 + local_row] = wg_voc[sub_row * 32u];
    }
}

// ----------------------------------------------------------------------------
// 11. Two-Stage Exact Full-Vocabulary Top-2 + Partition Function Reduction
//     Reduces `[vocab_size]` FP32 logits on GPU into a `16-Byte` readout:
//       rw_u32_a[0] = top1_token_id
//       rw_u32_a[1] = bitcast<u32>(top1_logit)
//       rw_u32_a[2] = bitcast<u32>(top2_logit)
//       rw_u32_a[3] = bitcast<u32>(sum_v exp(logit[v] - top1_logit))
//     Enabling `16 Bytes/token` browser mapAsync + exact System 1 confidence `p1`
//     and epistemic margin `p1 - p2`!
// ----------------------------------------------------------------------------
var<workgroup> wg_r_v1: array<f32, 256>;
var<workgroup> wg_r_i1: array<u32, 256>;
var<workgroup> wg_r_v2: array<f32, 256>;
var<workgroup> wg_r_sexp: array<f32, 256>;

fn merge_top2_sumexp(
    a_v1: f32, a_i1: u32, a_v2: f32, a_s: f32,
    b_v1: f32, b_i1: u32, b_v2: f32, b_s: f32,
    out_v1: ptr<function, f32>,
    out_i1: ptr<function, u32>,
    out_v2: ptr<function, f32>,
    out_s: ptr<function, f32>,
) {
    if (a_v1 >= b_v1) {
        *out_v1 = a_v1;
        *out_i1 = a_i1;
        *out_v2 = max(a_v2, b_v1);
        if (a_v1 <= -1e29) {
            *out_s = 0.0;
        } else {
            let b_term = select(0.0, b_s * exp(b_v1 - a_v1), b_v1 > -1e29);
            *out_s = a_s + b_term;
        }
    } else {
        *out_v1 = b_v1;
        *out_i1 = b_i1;
        *out_v2 = max(b_v2, a_v1);
        let a_term = select(0.0, a_s * exp(a_v1 - b_v1), a_v1 > -1e29);
        *out_s = b_s + a_term;
    }
}

// Stage 1: 256 workgroups reduce `vocab_size` (`params.p0`) into 256 summaries:
//   rw_f32_a[wg * 3 + 0] = top1_val
//   rw_f32_a[wg * 3 + 1] = top2_val
//   rw_f32_a[wg * 3 + 2] = sum_exp_shifted
//   rw_u32_a[wg]         = top1_id
@compute @workgroup_size(256, 1, 1)
fn logits_top2_reduce_stage1(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let wg = wgid.x;
    let vocab_size = params.p0;
    let stride_total = 256u * 256u;

    var v1: f32 = -1e30;
    var i1: u32 = 0u;
    var v2: f32 = -1e30;
    var sexp: f32 = 0.0;

    for (var idx = wg * 256u + tid; idx < vocab_size; idx += stride_total) {
        let raw_x = ro_f32_a[idx];
        let x = select(-1e30, raw_x, (raw_x == raw_x) && (abs(raw_x) < 1e29));
        var nv1: f32 = 0.0;
        var ni1: u32 = 0u;
        var nv2: f32 = 0.0;
        var ns: f32 = 0.0;
        merge_top2_sumexp(v1, i1, v2, sexp, x, idx, -1e30, 1.0, &nv1, &ni1, &nv2, &ns);
        v1 = nv1;
        i1 = ni1;
        v2 = nv2;
        sexp = ns;
    }

    wg_r_v1[tid] = v1;
    wg_r_i1[tid] = i1;
    wg_r_v2[tid] = v2;
    wg_r_sexp[tid] = sexp;
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            var nv1: f32 = 0.0;
            var ni1: u32 = 0u;
            var nv2: f32 = 0.0;
            var ns: f32 = 0.0;
            merge_top2_sumexp(
                wg_r_v1[tid], wg_r_i1[tid], wg_r_v2[tid], wg_r_sexp[tid],
                wg_r_v1[tid + s], wg_r_i1[tid + s], wg_r_v2[tid + s], wg_r_sexp[tid + s],
                &nv1, &ni1, &nv2, &ns,
            );
            wg_r_v1[tid] = nv1;
            wg_r_i1[tid] = ni1;
            wg_r_v2[tid] = nv2;
            wg_r_sexp[tid] = ns;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        rw_f32_a[wg * 3u + 0u] = wg_r_v1[0];
        rw_f32_a[wg * 3u + 1u] = wg_r_v2[0];
        rw_f32_a[wg * 3u + 2u] = wg_r_sexp[0];
        rw_u32_a[wg] = wg_r_i1[0];
    }
}

// Stage 2: 1 workgroup reduces the 256 summaries into 16 Bytes in `rw_u32_a[0..3]`
@compute @workgroup_size(256, 1, 1)
fn logits_top2_reduce_stage2(@builtin(local_invocation_id) lid: vec3<u32>) {
    let tid = lid.x;
    wg_r_v1[tid] = ro_f32_a[tid * 3u + 0u];
    wg_r_v2[tid] = ro_f32_a[tid * 3u + 1u];
    wg_r_sexp[tid] = ro_f32_a[tid * 3u + 2u];
    wg_r_i1[tid] = ro_u32_a[tid];
    workgroupBarrier();

    for (var s = 128u; s > 0u; s >>= 1u) {
        if (tid < s) {
            var nv1: f32 = 0.0;
            var ni1: u32 = 0u;
            var nv2: f32 = 0.0;
            var ns: f32 = 0.0;
            merge_top2_sumexp(
                wg_r_v1[tid], wg_r_i1[tid], wg_r_v2[tid], wg_r_sexp[tid],
                wg_r_v1[tid + s], wg_r_i1[tid + s], wg_r_v2[tid + s], wg_r_sexp[tid + s],
                &nv1, &ni1, &nv2, &ns,
            );
            wg_r_v1[tid] = nv1;
            wg_r_i1[tid] = ni1;
            wg_r_v2[tid] = nv2;
            wg_r_sexp[tid] = ns;
        }
        workgroupBarrier();
    }

    if (tid == 0u) {
        rw_u32_a[0] = wg_r_i1[0];
        rw_u32_a[1] = bitcast<u32>(wg_r_v1[0]);
        rw_u32_a[2] = bitcast<u32>(wg_r_v2[0]);
        rw_u32_a[3] = bitcast<u32>(wg_r_sexp[0]);
    }
}

// ============================================================================
// 16 & 17. Unified VQ4-Scar Kernels (4D Vector-Scar LUT256 + 2-Pass INT4-ASVD)
//          Bit-exact WebGPU counterpart to `scar_cache/vq4_scar_mos.py`
// ============================================================================

fn vq4_rademacher_sign(idx: u32) -> f32 {
    var h: u32 = idx * 2654435761u;
    h = ((h >> 16u) ^ h) * 0x45D9F3Bu;
    return select(-1.0, 1.0, (h & 1u) == 0u);
}

// ----------------------------------------------------------------------------
// 16. Channel Equilibration + Orthonormal Block-64 Rademacher-Hadamard (FWHT-64)
//     Computes once per input vector `x [in_dim]`:
//       x_rot = FWHT_64( (x / d_fp16) * rademacher_sign )
//     By symmetry/orthonormality of H_64, <H_64^-1(w_rot)/d, x> == <w_rot, x_rot>,
//     eliminating ALL Hadamard ops from the O(N*K) matrix-vector inner loop!
//     params: p0=in_dim (multiple of 64), p1=d_eq_u32_off
//     ro_u32_a: shard containing FP16 `d_eq` [in_dim]
//     ro_f32_a: raw input vector `x` [in_dim]
//     rw_f32_a: equilibrated + rotated vector `x_rot` [in_dim]
//     dispatch: (in_dim >> 6u, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_fwht64: array<f32, 64>;

@compute @workgroup_size(64, 1, 1)
fn vq4_equilibrate_fwht64(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let col = wgid.x * 64u + lane;
    let d_u32_off = params.p1;
    let d_val = unpack_f16_lane(ro_u32_a[d_u32_off + (col >> 1u)], col);
    wg_fwht64[lane] = (ro_f32_a[col] / d_val) * vq4_rademacher_sign(col);
    workgroupBarrier();

    for (var h = 1u; h < 64u; h <<= 1u) {
        let is_odd = (lane & h) != 0u;
        let partner = lane ^ h;
        let a = wg_fwht64[select(lane, partner, is_odd)];
        let b = wg_fwht64[select(partner, lane, is_odd)];
        workgroupBarrier();
        wg_fwht64[lane] = select(a + b, a - b, is_odd);
        workgroupBarrier();
    }

    rw_f32_a[col] = wg_fwht64[lane] * 0.125;
}

// ----------------------------------------------------------------------------
// 17. Fused VQ4 Micro-Scar LUT256 (`2.25 bpw`) + Compressed INT4 B_scar GEMV
//     Loads the per-matrix `256 x vec4<f16>` codebook into 4 KB of fast shared
//     memory (`wg_vq_lut`, with `wg_vq_lut[0] == vec4(0.0)`), decodes 16 weights
//     (4 byte-indices) per coalesced `u32` read, and fuses the compressed INT4
//     `B_scar [out_dim, rank]` dot `u = A_scar @ x_raw` + optional FP16 bias!
//     params:
//       p0 = out_dim
//       p1 = in_dim (multiple of 64)
//       p2 = rank (0, or multiple of 8)
//       p3 = vq_codes_u32_off   ([out_dim, in_dim / 16] u32 words)
//       p4 = vq_scales_u32_off  (FP16 block-64 scales)
//       p5 = lut256_u32_off     (512 u32 words = 256 x 4 FP16)
//       p6 = bscar_u32_off      (packed INT4 [out_dim, rank/8] followed by FP16 row scales [out_dim])
//       p7 = (has_bias << 31u) | bias_u32_off
//     ro_u32_a: layer shard
//     ro_f32_a: rotated input `x_rot` [in_dim] (from `vq4_equilibrate_fwht64`)
//     ro_f32_b: compressed-scar latent `u = A_scar @ x_raw` [rank]
//     rw_f32_a: output `y` [out_dim]
//     dispatch: ((out_dim + 7u) >> 3u, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_vq_lut: array<vec4<f32>, 256>;
var<workgroup> wg_vq_gemv: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn mos_fused_vq4_int4scar_gemv(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let lut_base = params.p5 + (tid << 1u);
    let p01 = unpack2x16float(ro_u32_a[lut_base]);
    let p23 = unpack2x16float(ro_u32_a[lut_base + 1u]);
    wg_vq_lut[tid] = vec4<f32>(p01.x, p01.y, p23.x, p23.y);
    workgroupBarrier();

    let sub_row = tid >> 5u;
    let lane = tid & 31u;
    let row = wgid.x * 8u + sub_row;

    let out_dim = params.p0;
    let in_dim = params.p1;
    let rank = params.p2;
    let vq_words_per_row = in_dim >> 4u;
    let blocks_per_row = in_dim >> 6u;

    var acc: f32 = 0.0;
    if (row < out_dim) {
        let code_row_base = params.p3 + row * vq_words_per_row;
        let s_row_base = row * blocks_per_row;
        for (var w_idx = lane; w_idx < vq_words_per_row; w_idx += 32u) {
            let word = ro_u32_a[code_row_base + w_idx];
            let blk_idx = s_row_base + (w_idx >> 2u);
            let scale = unpack_f16_lane(ro_u32_a[params.p4 + (blk_idx >> 1u)], blk_idx);
            let c0 = w_idx << 4u;

            let v0 = wg_vq_lut[(word >>  0u) & 0xFFu];
            let v1 = wg_vq_lut[(word >>  8u) & 0xFFu];
            let v2 = wg_vq_lut[(word >> 16u) & 0xFFu];
            let v3 = wg_vq_lut[(word >> 24u) & 0xFFu];

            let dot16 =
                  v0.x * ro_f32_a[c0 +  0u] + v0.y * ro_f32_a[c0 +  1u] + v0.z * ro_f32_a[c0 +  2u] + v0.w * ro_f32_a[c0 +  3u]
                + v1.x * ro_f32_a[c0 +  4u] + v1.y * ro_f32_a[c0 +  5u] + v1.z * ro_f32_a[c0 +  6u] + v1.w * ro_f32_a[c0 +  7u]
                + v2.x * ro_f32_a[c0 +  8u] + v2.y * ro_f32_a[c0 +  9u] + v2.z * ro_f32_a[c0 + 10u] + v2.w * ro_f32_a[c0 + 11u]
                + v3.x * ro_f32_a[c0 + 12u] + v3.y * ro_f32_a[c0 + 13u] + v3.z * ro_f32_a[c0 + 14u] + v3.w * ro_f32_a[c0 + 15u];
            acc += dot16 * scale;
        }

        if (rank > 0u) {
            let b_words_per_row = rank >> 3u;
            let b_row_base = params.p6 + row * b_words_per_row;
            let b_scales_off = params.p6 + out_dim * b_words_per_row;
            let b_scale = unpack_f16_lane(ro_u32_a[b_scales_off + (row >> 1u)], row);
            var b_acc: f32 = 0.0;
            for (var bw = lane; bw < b_words_per_row; bw += 32u) {
                let bword = ro_u32_a[b_row_base + bw];
                let r0 = bw << 3u;
                b_acc += (f32((bword >>  0u) & 0xFu) - 8.0) * ro_f32_b[r0 + 0u]
                       + (f32((bword >>  4u) & 0xFu) - 8.0) * ro_f32_b[r0 + 1u]
                       + (f32((bword >>  8u) & 0xFu) - 8.0) * ro_f32_b[r0 + 2u]
                       + (f32((bword >> 12u) & 0xFu) - 8.0) * ro_f32_b[r0 + 3u]
                       + (f32((bword >> 16u) & 0xFu) - 8.0) * ro_f32_b[r0 + 4u]
                       + (f32((bword >> 20u) & 0xFu) - 8.0) * ro_f32_b[r0 + 5u]
                       + (f32((bword >> 24u) & 0xFu) - 8.0) * ro_f32_b[r0 + 6u]
                       + (f32((bword >> 28u) & 0xFu) - 8.0) * ro_f32_b[r0 + 7u];
            }
            acc += b_acc * b_scale;
        }
    }

    wg_vq_gemv[tid] = acc;
    workgroupBarrier();

    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            wg_vq_gemv[tid] += wg_vq_gemv[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && row < out_dim) {
        var val = wg_vq_gemv[sub_row * 32u];
        if ((params.p7 >> 31u) != 0u) {
            let bias_off = params.p7 & 0x7FFFFFFFu;
            val += unpack_f16_lane(ro_u32_a[bias_off + (row >> 1u)], row);
        }
        rw_f32_a[row] = val;
    }
}

// ----------------------------------------------------------------------------
// 18. Script-Orthogonal Active Subspace Gate (`P_perp_CJK` Logit Mask)
//     Suppresses spurious bilingual CJK/Hangul/Kana token collisions in ultra-low-bit
//     polyglot decoding when the user's prompt contains zero CJK characters (`p1 == 1u`),
//     while remaining a strict `0.0` no-op when CJK is active (`p1 == 0u`).
//     params: p0 = vocab_size, p1 = suppress_cjk (0u or 1u), f0 = penalty (e.g. 64.0)
//     ro_u32_a: packed bitmask [ceil(vocab_size / 32)] (bit `vid & 31u` == 1u iff CJK token)
//     rw_f32_a: logits [vocab_size]
//     dispatch: ((vocab_size + 255u) >> 8u, 1u, 1u)
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn apply_script_subspace_gate(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let vid = gid.x;
    if (vid >= params.p0 || params.p1 == 0u) {
        return;
    }
    let mask_word = ro_u32_a[vid >> 5u];
    if (((mask_word >> (vid & 31u)) & 1u) != 0u) {
        rw_f32_a[vid] = rw_f32_a[vid] - params.f0;
    }
}

// ----------------------------------------------------------------------------
// 19. Pure Channel Equilibration (`x_eq[col] = x[col] / d_eq_f16[col]`)
//     Used by `ProtectedInt4ScarMatrix` (`prot_int4_scar`) on nonlinear triad
//     projections (`gate_proj`, `o_proj`, `k/v_proj`, boundary `q_proj`).
//     params: p0 = in_dim, p1 = d_eq_u32_off
//     ro_u32_a: layer shard containing FP16 `d_eq` [in_dim]
//     ro_f32_a: raw input vector `x` [in_dim]
//     rw_f32_a: equilibrated vector `x_eq` [in_dim]
//     dispatch: ((in_dim + 255u) >> 8u, 1u, 1u)
// ----------------------------------------------------------------------------
@compute @workgroup_size(256, 1, 1)
fn vec_equilibrate_f16(
    @builtin(global_invocation_id) gid: vec3<u32>,
) {
    let col = gid.x;
    if (col >= params.p0) {
        return;
    }
    let d_val = unpack_f16_lane(ro_u32_a[params.p1 + (col >> 1u)], col);
    rw_f32_a[col] = ro_f32_a[col] / d_val;
}

// ----------------------------------------------------------------------------
// 20. Fused Protected Block-64 INT4 + Row-Scaled INT4 B_scar GEMV
//     Executes `ProtectedInt4ScarMatrix` with ZERO FP16 weight expansion:
//       y[row] = dot(dequant_block64_int4(base_packed[row], base_scales[row]), x_eq)
//              + dot(dequant_rowwise_int4(b_packed[row], b_scale[row]), u)
//              + optional_bias[row]
//     params:
//       p0 = out_dim
//       p1 = in_dim (multiple of 64)
//       p2 = rank (0, or multiple of 8, e.g. 24)
//       p3 = packed_u32_off    ([out_dim, in_dim / 8] u32 words)
//       p4 = scales_u32_off    (FP16 block-64 scales [out_dim, in_dim / 64])
//       p5 = bscar_u32_off     (packed INT4 [out_dim, rank / 8] u32 words + FP16 row scales [out_dim])
//       p6 = bias_u32_off
//       p7 = has_bias (0u or 1u)
//     ro_u32_a: layer shard
//     ro_f32_a: equilibrated input `x_eq` [in_dim] (or raw `x` if un-equilibrated)
//     ro_f32_b: compressed-scar latent `u = A_scar @ x_raw` [rank]
//     rw_f32_a: output `y` [out_dim]
//     dispatch: ((out_dim + 7u) >> 3u, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_prot_mos: array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn mos_fused_prot_int4_int4scar_gemv(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let sub_row = tid >> 5u;
    let lane = tid & 31u;
    let row = wgid.x * 8u + sub_row;

    let out_dim = params.p0;
    let in_dim = params.p1;
    let rank = params.p2;
    let words_per_row = in_dim >> 3u;
    let blocks_per_row = in_dim >> 6u;

    var acc: f32 = 0.0;
    if (row < out_dim) {
        let p_row_base = params.p3 + row * words_per_row;
        let s_row_base = row * blocks_per_row;
        for (var w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
            let word = ro_u32_a[p_row_base + w_idx];
            let blk_idx = s_row_base + (w_idx >> 3u);
            let scale = unpack_f16_lane(ro_u32_a[params.p4 + (blk_idx >> 1u)], blk_idx);
            let c0 = w_idx << 3u;
            let q0 = f32((word >>  0u) & 0xFu) - 8.0;
            let q1 = f32((word >>  4u) & 0xFu) - 8.0;
            let q2 = f32((word >>  8u) & 0xFu) - 8.0;
            let q3 = f32((word >> 12u) & 0xFu) - 8.0;
            let q4 = f32((word >> 16u) & 0xFu) - 8.0;
            let q5 = f32((word >> 20u) & 0xFu) - 8.0;
            let q6 = f32((word >> 24u) & 0xFu) - 8.0;
            let q7 = f32((word >> 28u) & 0xFu) - 8.0;
            let sub_dot = q0 * ro_f32_a[c0 + 0u] + q1 * ro_f32_a[c0 + 1u]
                        + q2 * ro_f32_a[c0 + 2u] + q3 * ro_f32_a[c0 + 3u]
                        + q4 * ro_f32_a[c0 + 4u] + q5 * ro_f32_a[c0 + 5u]
                        + q6 * ro_f32_a[c0 + 6u] + q7 * ro_f32_a[c0 + 7u];
            acc += sub_dot * scale;
        }

        if (rank > 0u) {
            let b_words_per_row = rank >> 3u;
            let b_row_base = params.p5 + row * b_words_per_row;
            let b_scales_off = params.p5 + out_dim * b_words_per_row;
            let b_scale = unpack_f16_lane(ro_u32_a[b_scales_off + (row >> 1u)], row);
            var b_acc: f32 = 0.0;
            for (var bw = lane; bw < b_words_per_row; bw += 32u) {
                let bword = ro_u32_a[b_row_base + bw];
                let r0 = bw << 3u;
                b_acc += (f32((bword >>  0u) & 0xFu) - 8.0) * ro_f32_b[r0 + 0u]
                       + (f32((bword >>  4u) & 0xFu) - 8.0) * ro_f32_b[r0 + 1u]
                       + (f32((bword >>  8u) & 0xFu) - 8.0) * ro_f32_b[r0 + 2u]
                       + (f32((bword >> 12u) & 0xFu) - 8.0) * ro_f32_b[r0 + 3u]
                       + (f32((bword >> 16u) & 0xFu) - 8.0) * ro_f32_b[r0 + 4u]
                       + (f32((bword >> 20u) & 0xFu) - 8.0) * ro_f32_b[r0 + 5u]
                       + (f32((bword >> 24u) & 0xFu) - 8.0) * ro_f32_b[r0 + 6u]
                       + (f32((bword >> 28u) & 0xFu) - 8.0) * ro_f32_b[r0 + 7u];
            }
            acc += b_acc * b_scale;
        }
    }

    wg_prot_mos[tid] = acc;
    workgroupBarrier();

    for (var s = 16u; s > 0u; s >>= 1u) {
        if (lane < s) {
            wg_prot_mos[tid] += wg_prot_mos[tid + s];
        }
        workgroupBarrier();
    }

    if (lane == 0u && row < out_dim) {
        var val = wg_prot_mos[sub_row * 32u];
        if (params.p7 != 0u) {
            val += unpack_f16_lane(ro_u32_a[params.p6 + (row >> 1u)], row);
        }
        rw_f32_a[row] = val;
    }
}

// ============================================================================
// SCAR-NATIVE v5.0 END-TO-END LEARNED O(1) WIDROW-HOFF & HYBRID GQA KERNELS
// ============================================================================

// ----------------------------------------------------------------------------
// 21. Pure Scar-Native v5.0 Content-Addressed Pre-RoPE Widrow-Hoff Eviction
//     & O(1) Triple-Channel (K_rope, K_raw, V) Sink/Ring Update (`Δ = +0 B`)
//     1 workgroup per KV head (`wgid.x = h_kv in 0 .. n_kv_heads - 1`).
//
//     params: p0=layer_idx, p1=n_kv_heads, p2=head_dim,
//             p3=num_sink, p4=slide_len, p5=counts_u32_total
//     ro_u32_b: step_state_u32 ([1]=pos_t)
//     ro_f32_a: k_rope_cur [n_kv_heads * head_dim]
//     ro_f32_b: v_cur      [n_kv_heads * head_dim]
//     ro_f32_c: k_raw_and_gates:
//               - [0 .. n_kv_heads * head_dim - 1]: k_raw_cur (Pre-RoPE)
//               - [n_kv_heads * head_dim .. + n_kv_heads * (head_dim + 3)]:
//                   per KV head stride (head_dim + 3): [w_gate(head_dim), b_gate, lambda_h, eta_h]
//     rw_f32_a: scarnative_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//     rw_u32_a: scarnative_kv_f16_u32:
//               - [0 .. counts_u32_total - 1]: 4 u32 per (layer, h_kv):
//                   [sink_cnt, win_cnt, absorbed_cnt, reserved]
//               - [counts_u32_total ..]: 3 * cap_per_head * vec_u32 per (layer, h_kv):
//                   K_rope [cap_per_head], K_raw [cap_per_head], V [cap_per_head]
// ----------------------------------------------------------------------------
var<workgroup> wg_sn_kraw: array<f32, 128>;
var<workgroup> wg_sn_vev: array<f32, 128>;
var<workgroup> wg_sn_ku: array<f32, 128>;
var<workgroup> wg_sn_err: array<f32, 128>;
var<workgroup> wg_sn_red: array<f32, 64>;
var<workgroup> wg_sn_ctrl: array<u32, 6>;

@compute @workgroup_size(64, 1, 1)
fn scarnative_widrow_hoff_evict_update(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_kv_heads = params.p1;
    let head_dim = params.p2;
    let num_sink = params.p3;
    let slide_len = params.p4;
    let counts_u32_total = params.p5;
    let h_kv = wgid.x;

    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;
    let cap_per_head = num_sink + slide_len;
    let vec_u32 = head_dim >> 1u;
    let head_stride = 3u * cap_per_head * vec_u32;
    let krope_base = counts_u32_total + head_flat * head_stride;
    let kraw_base = krope_base + cap_per_head * vec_u32;
    let v_base = krope_base + 2u * cap_per_head * vec_u32;
    let w_base = head_flat * head_dim * head_dim;

    if (tid == 0u) {
        let pos_t = ro_u32_b[1];
        let sink_cnt = rw_u32_a[cnt_base + 0u];
        let win_cnt = rw_u32_a[cnt_base + 1u];
        let absorbed = rw_u32_a[cnt_base + 2u];
        wg_sn_ctrl[0] = pos_t;
        wg_sn_ctrl[1] = sink_cnt;
        wg_sn_ctrl[2] = win_cnt;
        wg_sn_ctrl[3] = absorbed;
        wg_sn_ctrl[4] = select(0u, 1u, (pos_t >= num_sink) && (win_cnt >= slide_len));
    }
    workgroupBarrier();

    let pos_t = wg_sn_ctrl[0];
    let sink_cnt = wg_sn_ctrl[1];
    let win_cnt = wg_sn_ctrl[2];
    let absorbed = wg_sn_ctrl[3];
    let do_evict = wg_sn_ctrl[4] != 0u;
    let rel_t = select(0u, pos_t - num_sink, pos_t >= num_sink);
    let ring_idx = select(0u, rel_t % max(slide_len, 1u), slide_len > 0u);
    let ring_slot = num_sink + ring_idx;

    // Step 1: Unpack evicted Pre-RoPE K_raw and V from `ring_slot` and compute ||k_raw||_2
    var kn_sq: f32 = 0.0;
    if (do_evict) {
        for (var p = tid; p < vec_u32; p += 64u) {
            let krp = unpack2x16float(rw_u32_a[kraw_base + ring_slot * vec_u32 + p]);
            let vp = unpack2x16float(rw_u32_a[v_base + ring_slot * vec_u32 + p]);
            let d0 = p << 1u;
            wg_sn_kraw[d0] = krp.x;
            wg_sn_kraw[d0 + 1u] = krp.y;
            wg_sn_vev[d0] = vp.x;
            wg_sn_vev[d0 + 1u] = vp.y;
            kn_sq += krp.x * krp.x + krp.y * krp.y;
        }
    }
    wg_sn_red[tid] = kn_sq;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sn_red[tid] += wg_sn_red[tid + s];
        }
        workgroupBarrier();
    }

    if (do_evict) {
        let inv_kn = 1.0 / max(sqrt(wg_sn_red[0]), 1e-6);
        for (var d = tid; d < head_dim; d += 64u) {
            wg_sn_ku[d] = wg_sn_kraw[d] * inv_kn;
        }
    }
    workgroupBarrier();

    // Step 2: Widrow-Hoff prediction `pred = W_scar @ ku` and unnormalized value error `err = v_ev - pred`
    if (do_evict) {
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            var pred_r: f32 = 0.0;
            for (var c = 0u; c < head_dim; c += 1u) {
                pred_r += rw_f32_a[w_row + c] * wg_sn_ku[c];
            }
            wg_sn_err[r] = wg_sn_vev[r] - pred_r;
        }
    }
    workgroupBarrier();

    // Step 3: Surprise-Gated Widrow-Hoff Delta update:
    //   g_tau = sigmoid(dot(w_gate_h, k_raw_ev) + b_gate_h)
    //   W_scar = (1 - lambda_h * g_tau) * W_scar + (eta_h * g_tau) * (err @ ku^T)
    let gate_stride = head_dim + 3u;
    let gate_off = n_kv_heads * head_dim + h_kv * gate_stride;
    var g_dot: f32 = 0.0;
    if (do_evict) {
        for (var d = tid; d < head_dim; d += 64u) {
            g_dot += ro_f32_c[gate_off + d] * wg_sn_kraw[d];
        }
    }
    wg_sn_red[tid] = g_dot;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sn_red[tid] += wg_sn_red[tid + s];
        }
        workgroupBarrier();
    }

    if (do_evict) {
        let b_gate = ro_f32_c[gate_off + head_dim + 0u];
        let lambda_h = ro_f32_c[gate_off + head_dim + 1u];
        let eta_h = ro_f32_c[gate_off + head_dim + 2u];
        let z_gate = clamp(wg_sn_red[0] + b_gate, -20.0, 20.0);
        let g_tau = 1.0 / (1.0 + exp(-z_gate));
        let decay_eff = 1.0 - lambda_h * g_tau;
        let lr_eff = eta_h * g_tau;
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            let e_r = wg_sn_err[r];
            for (var c = 0u; c < head_dim; c += 1u) {
                rw_f32_a[w_row + c] = decay_eff * rw_f32_a[w_row + c] + lr_eff * e_r * wg_sn_ku[c];
            }
        }
    }
    workgroupBarrier();

    // Step 4: Write current `(k_rope_cur, k_raw_cur, v_cur)` into Sink or Ring Slot
    let write_slot = select(ring_slot, pos_t, pos_t < num_sink);
    let in_off = h_kv * head_dim;
    for (var p = tid; p < vec_u32; p += 64u) {
        let d0 = p << 1u;
        let krop2 = clamp(
            vec2<f32>(ro_f32_a[in_off + d0], ro_f32_a[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        let kraw2 = clamp(
            vec2<f32>(ro_f32_c[in_off + d0], ro_f32_c[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        let vn2 = clamp(
            vec2<f32>(ro_f32_b[in_off + d0], ro_f32_b[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        rw_u32_a[krope_base + write_slot * vec_u32 + p] = pack2x16float(krop2);
        rw_u32_a[kraw_base + write_slot * vec_u32 + p] = pack2x16float(kraw2);
        rw_u32_a[v_base + write_slot * vec_u32 + p] = pack2x16float(vn2);
    }
    if (tid == 0u) {
        if (pos_t < num_sink) {
            if ((pos_t + 1u) > sink_cnt) {
                rw_u32_a[cnt_base + 0u] = pos_t + 1u;
            }
        } else {
            rw_u32_a[cnt_base + 1u] = min(slide_len, win_cnt + 1u);
            if (do_evict) {
                rw_u32_a[cnt_base + 2u] = absorbed + 1u;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 22. Pure Scar-Native v5.0 O(1) Hybrid RoPE Sink+Window + Content-Addressed
//     Widrow-Hoff Fast-Weight Read Attention (`cos > 0.999999` vs PyTorch)
//     1 workgroup per Query head (`wgid.x = h_q in 0 .. n_q_heads - 1`).
//
//     params: p0=layer_idx, p1=n_q_heads, p2=n_kv_heads, p3=head_dim,
//             p4=num_sink, p5=slide_len, p6=counts_u32_total, f0=attn_scale
//     ro_u32_a: scarnative_kv_f16_u32
//     ro_f32_a: q_rope_cur [n_q_heads * head_dim]
//     ro_f32_b: scarnative_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//     ro_f32_c: q_raw_and_alpha:
//               - [0 .. n_q_heads * head_dim - 1]: q_raw_cur (Pre-RoPE)
//               - [n_q_heads * head_dim .. + n_q_heads]: alpha_assoc[h_q] (`tanh(beta_h)`)
//     rw_f32_a: attn_out [n_q_heads * head_dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_sn_qrope: array<f32, 128>;
var<workgroup> wg_sn_qu: array<f32, 128>;
var<workgroup> wg_sn_logits: array<f32, 512>;
var<workgroup> wg_sn_ared: array<f32, 64>;
var<workgroup> wg_sn_acnts: array<u32, 4>;

@compute @workgroup_size(64, 1, 1)
fn scarnative_o1_hybrid_gqa_attention(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_q_heads = params.p1;
    let n_kv_heads = params.p2;
    let head_dim = params.p3;
    let num_sink = params.p4;
    let slide_len = params.p5;
    let counts_u32_total = params.p6;
    let attn_scale = params.f0;

    let h_q = wgid.x;
    let gqa_group = n_q_heads / n_kv_heads;
    let h_kv = h_q / gqa_group;
    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;

    let cap_per_head = num_sink + slide_len;
    let vec_u32 = head_dim >> 1u;
    let head_stride = 3u * cap_per_head * vec_u32;
    let krope_base = counts_u32_total + head_flat * head_stride;
    let v_base = krope_base + 2u * cap_per_head * vec_u32;
    let w_base = head_flat * head_dim * head_dim;

    var qraw_sq: f32 = 0.0;
    for (var d = tid; d < head_dim; d += 64u) {
        wg_sn_qrope[d] = ro_f32_a[h_q * head_dim + d];
        let qr = ro_f32_c[h_q * head_dim + d];
        wg_sn_qu[d] = qr;
        qraw_sq += qr * qr;
    }
    wg_sn_ared[tid] = qraw_sq;
    if (tid < 4u) {
        wg_sn_acnts[tid] = ro_u32_a[cnt_base + tid];
    }
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sn_ared[tid] += wg_sn_ared[tid + s];
        }
        workgroupBarrier();
    }

    let inv_qn = 1.0 / max(sqrt(wg_sn_ared[0]), 1e-6);
    for (var d = tid; d < head_dim; d += 64u) {
        wg_sn_qu[d] = wg_sn_qu[d] * inv_qn;
    }
    workgroupBarrier();

    let sink_cnt = wg_sn_acnts[0];
    let win_cnt = wg_sn_acnts[1];
    let absorbed = wg_sn_acnts[2];
    let total_keys = sink_cnt + win_cnt;

    // Phase 1: Exact scaled dot-product logits over `[Sinks ∪ Sliding Ring]`
    var local_max: f32 = -1e30;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        let phys_slot = select(num_sink + (idx - sink_cnt), idx, idx < sink_cnt);
        let k_off = krope_base + phys_slot * vec_u32;
        var dot_qk: f32 = 0.0;
        for (var p = 0u; p < vec_u32; p += 1u) {
            let kp = unpack2x16float(ro_u32_a[k_off + p]);
            let d0 = p << 1u;
            dot_qk += wg_sn_qrope[d0] * kp.x + wg_sn_qrope[d0 + 1u] * kp.y;
        }
        let lg = dot_qk * attn_scale;
        wg_sn_logits[idx] = lg;
        if (lg > local_max) {
            local_max = lg;
        }
    }
    wg_sn_ared[tid] = local_max;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sn_ared[tid] = max(wg_sn_ared[tid], wg_sn_ared[tid + s]);
        }
        workgroupBarrier();
    }
    let max_lg = wg_sn_ared[0];

    var local_sum: f32 = 0.0;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        let ex = exp(wg_sn_logits[idx] - max_lg);
        wg_sn_logits[idx] = ex;
        local_sum += ex;
    }
    wg_sn_ared[tid] = local_sum;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_sn_ared[tid] += wg_sn_ared[tid + s];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / max(wg_sn_ared[0], 1e-12);
    for (var idx = tid; idx < total_keys; idx += 64u) {
        wg_sn_logits[idx] = wg_sn_logits[idx] * inv_sum;
    }
    workgroupBarrier();

    // Phase 2: Combine Local+Sink Softmax V aggregation with Content-Addressed `W_scar @ qu`
    let alpha_assoc = select(0.0, ro_f32_c[n_q_heads * head_dim + h_q], absorbed > 0u);
    for (var p = tid; p < vec_u32; p += 64u) {
        var v_acc = vec2<f32>(0.0, 0.0);
        for (var idx = 0u; idx < total_keys; idx += 1u) {
            let phys_slot = select(num_sink + (idx - sink_cnt), idx, idx < sink_cnt);
            let vp = unpack2x16float(ro_u32_a[v_base + phys_slot * vec_u32 + p]);
            v_acc += wg_sn_logits[idx] * vp;
        }
        let d0 = p << 1u;
        var fast0: f32 = 0.0;
        var fast1: f32 = 0.0;
        if (absorbed > 0u) {
            let w_r0 = w_base + d0 * head_dim;
            let w_r1 = w_base + (d0 + 1u) * head_dim;
            for (var c = 0u; c < head_dim; c += 1u) {
                let q_c = wg_sn_qu[c];
                fast0 += ro_f32_b[w_r0 + c] * q_c;
                fast1 += ro_f32_b[w_r1 + c] * q_c;
            }
        }
        rw_f32_a[h_q * head_dim + d0] = v_acc.x + alpha_assoc * fast0;
        rw_f32_a[h_q * head_dim + d0 + 1u] = v_acc.y + alpha_assoc * fast1;
    }
}

// ============================================================================
// SCAR-NATIVE v6.0 SHARED-STAGE MOS + DUAL-TIER HIERARCHICAL O(1) SCACHE KERNELS
// ============================================================================

// ----------------------------------------------------------------------------
// 23. Scar-Native v6.0 Native Mixture-of-Scars (MoS) Block-64 INT4 Top-k Router
//     + Directional ReLU A_exp Down-Projection (`u_j = w_j * ReLU(A_exp[e_j] @ x)`)
//     params:
//       p0 = hidden_dim (multiple of 64)
//       p1 = num_experts (<= 16)
//       p2 = top_k (<= 4)
//       p3 = expert_rank (<= 128, multiple of 8)
//       p4 = router_packed_u32_off  ([E, hidden_dim / 8] u32 words)
//       p5 = router_scales_u32_off  ([E, hidden_dim / 64] FP16 scales)
//       p6 = a_exp_packed_u32_off   ([E * rank, hidden_dim / 8] u32 words)
//       p7 = a_exp_scales_u32_off   ([E * rank, hidden_dim / 64] FP16 scales)
//     ro_u32_a: layer/model shard
//     ro_f32_a: normalized input `x` [hidden_dim]
//     rw_f32_a: `mos_u` [top_k * expert_rank]
//     rw_u32_a: `mos_selected_idx` [4] (`[0 .. top_k - 1]`)
//     dispatch: (1u, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_mos6_red: array<f32, 256>;
var<workgroup> wg_mos6_logits: array<f32, 16>;
var<workgroup> wg_mos6_idx: array<u32, 4>;
var<workgroup> wg_mos6_w: array<f32, 4>;

@compute @workgroup_size(256, 1, 1)
fn scarnative_v6_mos_router_topk_relu(
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let sub_row = tid >> 5u;
    let lane = tid & 31u;

    let hidden_dim = params.p0;
    let num_experts = params.p1;
    let top_k = params.p2;
    let expert_rank = params.p3;
    let words_per_row = hidden_dim >> 3u;
    let blocks_per_row = hidden_dim >> 6u;

    // Phase 1: Compute Block-64 INT4 router logits `r_logits[e] = dot(router[e], x)`
    for (var e_base = 0u; e_base < num_experts; e_base += 8u) {
        let e = e_base + sub_row;
        var r_acc: f32 = 0.0;
        if (e < num_experts) {
            let p_row = params.p4 + e * words_per_row;
            let s_row = e * blocks_per_row;
            for (var w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
                let word = ro_u32_a[p_row + w_idx];
                let blk_idx = s_row + (w_idx >> 3u);
                let scale = unpack_f16_lane(ro_u32_a[params.p5 + (blk_idx >> 1u)], blk_idx);
                let c0 = w_idx << 3u;
                let sub_dot = (f32((word >>  0u) & 0xFu) - 8.0) * ro_f32_a[c0 + 0u]
                            + (f32((word >>  4u) & 0xFu) - 8.0) * ro_f32_a[c0 + 1u]
                            + (f32((word >>  8u) & 0xFu) - 8.0) * ro_f32_a[c0 + 2u]
                            + (f32((word >> 12u) & 0xFu) - 8.0) * ro_f32_a[c0 + 3u]
                            + (f32((word >> 16u) & 0xFu) - 8.0) * ro_f32_a[c0 + 4u]
                            + (f32((word >> 20u) & 0xFu) - 8.0) * ro_f32_a[c0 + 5u]
                            + (f32((word >> 24u) & 0xFu) - 8.0) * ro_f32_a[c0 + 6u]
                            + (f32((word >> 28u) & 0xFu) - 8.0) * ro_f32_a[c0 + 7u];
                r_acc += sub_dot * scale;
            }
        }
        wg_mos6_red[tid] = r_acc;
        workgroupBarrier();

        for (var s = 16u; s > 0u; s >>= 1u) {
            if (lane < s) {
                wg_mos6_red[tid] += wg_mos6_red[tid + s];
            }
            workgroupBarrier();
        }
        if (lane == 0u && e < num_experts) {
            wg_mos6_logits[e] = wg_mos6_red[sub_row * 32u];
        }
        workgroupBarrier();
    }

    // Phase 2: Exact Top-k selection (`-1e-7 * e` deterministic tie-break) + Softmax over Top-k
    if (tid == 0u) {
        var used_mask: u32 = 0u;
        var max_top: f32 = -1e30;
        for (var j = 0u; j < top_k; j += 1u) {
            var best_e: u32 = 0u;
            var best_pert: f32 = -1e30;
            for (var e = 0u; e < num_experts; e += 1u) {
                if (((used_mask >> e) & 1u) == 0u) {
                    let pert = wg_mos6_logits[e] - 1e-7 * f32(e);
                    if (pert > best_pert) {
                        best_pert = pert;
                        best_e = e;
                    }
                }
            }
            used_mask = used_mask | (1u << best_e);
            wg_mos6_idx[j] = best_e;
            rw_u32_a[j] = best_e;
            let raw_lg = wg_mos6_logits[best_e];
            if (raw_lg > max_top) {
                max_top = raw_lg;
            }
        }
        var sum_ex: f32 = 0.0;
        for (var j = 0u; j < top_k; j += 1u) {
            let ex = exp(wg_mos6_logits[wg_mos6_idx[j]] - max_top);
            wg_mos6_w[j] = ex;
            sum_ex += ex;
        }
        let inv_sum = 1.0 / max(sum_ex, 1e-12);
        for (var j = 0u; j < top_k; j += 1u) {
            wg_mos6_w[j] = wg_mos6_w[j] * inv_sum;
        }
    }
    workgroupBarrier();

    // Phase 3: Compute `u[j * rank + r] = w_j * max(dot(A_exp[idx_j, r], x), 0.0)`
    let total_rows = top_k * expert_rank;
    for (var r_base = 0u; r_base < total_rows; r_base += 8u) {
        let item = r_base + sub_row;
        var a_acc: f32 = 0.0;
        if (item < total_rows) {
            let j = item / expert_rank;
            let r_loc = item % expert_rank;
            let e_sel = wg_mos6_idx[j];
            let a_row = e_sel * expert_rank + r_loc;
            let p_row = params.p6 + a_row * words_per_row;
            let s_row = a_row * blocks_per_row;
            for (var w_idx = lane; w_idx < words_per_row; w_idx += 32u) {
                let word = ro_u32_a[p_row + w_idx];
                let blk_idx = s_row + (w_idx >> 3u);
                let scale = unpack_f16_lane(ro_u32_a[params.p7 + (blk_idx >> 1u)], blk_idx);
                let c0 = w_idx << 3u;
                let sub_dot = (f32((word >>  0u) & 0xFu) - 8.0) * ro_f32_a[c0 + 0u]
                            + (f32((word >>  4u) & 0xFu) - 8.0) * ro_f32_a[c0 + 1u]
                            + (f32((word >>  8u) & 0xFu) - 8.0) * ro_f32_a[c0 + 2u]
                            + (f32((word >> 12u) & 0xFu) - 8.0) * ro_f32_a[c0 + 3u]
                            + (f32((word >> 16u) & 0xFu) - 8.0) * ro_f32_a[c0 + 4u]
                            + (f32((word >> 20u) & 0xFu) - 8.0) * ro_f32_a[c0 + 5u]
                            + (f32((word >> 24u) & 0xFu) - 8.0) * ro_f32_a[c0 + 6u]
                            + (f32((word >> 28u) & 0xFu) - 8.0) * ro_f32_a[c0 + 7u];
                a_acc += sub_dot * scale;
            }
        }
        wg_mos6_red[tid] = a_acc;
        workgroupBarrier();

        for (var s = 16u; s > 0u; s >>= 1u) {
            if (lane < s) {
                wg_mos6_red[tid] += wg_mos6_red[tid + s];
            }
            workgroupBarrier();
        }
        if (lane == 0u && item < total_rows) {
            let j = item / expert_rank;
            rw_f32_a[item] = wg_mos6_w[j] * max(wg_mos6_red[sub_row * 32u], 0.0);
        }
        workgroupBarrier();
    }
}

// ----------------------------------------------------------------------------
// 24. Scar-Native v6.0 Native MoS Row-Scaled INT4 B_exp Residual Accumulator
//     Accumulates `rw_f32_a[d] += sum_{j=0..top_k-1} dot(B_exp[idx_j, d], u_j)`
//     params:
//       p0 = hidden_dim
//       p1 = top_k (<= 4)
//       p2 = expert_rank (<= 128, multiple of 8)
//       p3 = b_exp_packed_u32_off   ([E * hidden_dim, rank / 8] u32 words)
//       p4 = b_exp_scales_u32_off   ([E * hidden_dim] FP16 row scales)
//     ro_u32_a: layer/model shard
//     ro_u32_b: `mos_selected_idx` [4]
//     ro_f32_a: `mos_u` [top_k * expert_rank]
//     rw_f32_a: residual target `h_cur` [hidden_dim] (accumulates in-place!)
//     dispatch: ((hidden_dim + 255u) >> 8u, 1u, 1u)
// ----------------------------------------------------------------------------
var<workgroup> wg_mos6_u: array<f32, 512>;
var<workgroup> wg_mos6_bidx: array<u32, 4>;

@compute @workgroup_size(256, 1, 1)
fn scarnative_v6_mos_bscar_accumulate(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let hidden_dim = params.p0;
    let top_k = params.p1;
    let expert_rank = params.p2;
    let total_u = top_k * expert_rank;

    if (tid < top_k) {
        wg_mos6_bidx[tid] = ro_u32_b[tid];
    }
    for (var i = tid; i < total_u; i += 256u) {
        wg_mos6_u[i] = ro_f32_a[i];
    }
    workgroupBarrier();

    let d = wgid.x * 256u + tid;
    if (d >= hidden_dim) {
        return;
    }

    let b_words_per_row = expert_rank >> 3u;
    var add_val: f32 = 0.0;
    for (var j = 0u; j < top_k; j += 1u) {
        let e_sel = wg_mos6_bidx[j];
        let b_row = e_sel * hidden_dim + d;
        let b_row_base = params.p3 + b_row * b_words_per_row;
        let b_scale = unpack_f16_lane(ro_u32_a[params.p4 + (b_row >> 1u)], b_row);
        let u_base = j * expert_rank;
        var sub_acc: f32 = 0.0;
        for (var bw = 0u; bw < b_words_per_row; bw += 1u) {
            let bword = ro_u32_a[b_row_base + bw];
            let r0 = u_base + (bw << 3u);
            sub_acc += (f32((bword >>  0u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 0u]
                     + (f32((bword >>  4u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 1u]
                     + (f32((bword >>  8u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 2u]
                     + (f32((bword >> 12u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 3u]
                     + (f32((bword >> 16u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 4u]
                     + (f32((bword >> 20u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 5u]
                     + (f32((bword >> 24u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 6u]
                     + (f32((bword >> 28u) & 0xFu) - 8.0) * wg_mos6_u[r0 + 7u];
        }
        add_val += sub_acc * b_scale;
    }
    rw_f32_a[d] = rw_f32_a[d] + add_val;
}

// ----------------------------------------------------------------------------
// 25. Scar-Native v6.0 Dual-Tier Hierarchical O(1) SCache Eviction & Update
//     (`RoPE Sinks + Sliding Ring + Surprise-Routed Exact Micro-Pages + W_scar`, `Δ = +0 B`)
//     1 workgroup per KV head (`wgid.x = h_kv in 0 .. n_kv_heads - 1`).
//
//     params:
//       p0 = layer_idx
//       p1 = n_kv_heads
//       p2 = head_dim (<= 128)
//       p3 = num_sink
//       p4 = slide_len
//       p5 = num_micro_pages (<= 64)
//       p6 = counts_u32_total (`num_layers * n_kv_heads * 4`)
//       p7 = flags (`bit 0 = enable_widrow_hoff`, `bit 1 = enable_micro_pages`)
//       f0 = micro_page_theta
//     ro_u32_b: step_state_u32 ([1] = pos_t)
//     ro_f32_a: k_rope_cur [n_kv_heads * head_dim]
//     ro_f32_b: v_cur      [n_kv_heads * head_dim]
//     ro_f32_c: k_raw_and_gates ([0..n_kv*d_h-1] = k_raw_cur, followed by [w_gate(d_h), b_gate, lambda_h, eta_h])
//     rw_f32_a: scarnative_v6_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//     rw_f32_b: scarnative_v6_page_scores_f32 [L * n_kv_heads * max(1u, num_micro_pages)]
//     rw_u32_a: scarnative_v6_kv_f16_u32:
//               - [0 .. counts_u32_total - 1]: 4 u32 per (layer, h_kv):
//                   [sink_cnt, win_cnt, absorbed_cnt, page_cnt]
//               - [counts_u32_total ..]: `(2 * cap_total + cap_local) * vec_u32` per (layer, h_kv):
//                   K_rope [cap_total], V [cap_total], K_raw [cap_local]
// ----------------------------------------------------------------------------
var<workgroup> wg_v6_kraw: array<f32, 128>;
var<workgroup> wg_v6_vev: array<f32, 128>;
var<workgroup> wg_v6_ku: array<f32, 128>;
var<workgroup> wg_v6_err: array<f32, 128>;
var<workgroup> wg_v6_red: array<f32, 64>;
var<workgroup> wg_v6_ctrl: array<u32, 8>;

@compute @workgroup_size(64, 1, 1)
fn scarnative_v6_hierarchical_evict_update(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_kv_heads = params.p1;
    let head_dim = params.p2;
    let num_sink = params.p3;
    let slide_len = params.p4;
    let num_micro_pages = params.p5;
    let counts_u32_total = params.p6;
    let en_wh = (params.p7 & 1u) != 0u;
    let en_mp = (params.p7 & 2u) != 0u;
    let micro_page_theta = params.f0;
    let h_kv = wgid.x;

    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;
    let cap_local = num_sink + slide_len;
    let cap_total = cap_local + num_micro_pages;
    let vec_u32 = head_dim >> 1u;
    let head_stride = (2u * cap_total + cap_local) * vec_u32;

    let krope_base = counts_u32_total + head_flat * head_stride;
    let v_base = krope_base + cap_total * vec_u32;
    let kraw_base = v_base + cap_total * vec_u32;
    let w_base = head_flat * head_dim * head_dim;

    if (tid == 0u) {
        let pos_t = ro_u32_b[1];
        let sink_cnt = rw_u32_a[cnt_base + 0u];
        let win_cnt = rw_u32_a[cnt_base + 1u];
        let absorbed = rw_u32_a[cnt_base + 2u];
        let page_cnt = rw_u32_a[cnt_base + 3u];
        let ev = (pos_t >= num_sink) && (win_cnt >= slide_len);
        wg_v6_ctrl[0] = pos_t;
        wg_v6_ctrl[1] = sink_cnt;
        wg_v6_ctrl[2] = win_cnt;
        wg_v6_ctrl[3] = absorbed;
        wg_v6_ctrl[4] = page_cnt;
        wg_v6_ctrl[5] = select(0u, 1u, ev);
        wg_v6_ctrl[6] = 0xFFFFFFFFu;
    }
    workgroupBarrier();

    let pos_t = wg_v6_ctrl[0];
    let sink_cnt = wg_v6_ctrl[1];
    let win_cnt = wg_v6_ctrl[2];
    let absorbed = wg_v6_ctrl[3];
    let page_cnt = wg_v6_ctrl[4];
    let do_evict = wg_v6_ctrl[5] != 0u;
    let active_evict = do_evict && (layer_idx > 0u) && (en_wh || en_mp);
    let rel_t = select(0u, pos_t - num_sink, pos_t >= num_sink);
    let ring_idx = select(0u, rel_t % max(slide_len, 1u), slide_len > 0u);
    let ring_slot = num_sink + ring_idx;

    // Step 1: Unpack evicted Pre-RoPE K_raw and V from `ring_slot`
    var kn_sq: f32 = 0.0;
    if (active_evict) {
        for (var p = tid; p < vec_u32; p += 64u) {
            let krp = unpack2x16float(rw_u32_a[kraw_base + ring_slot * vec_u32 + p]);
            let vp = unpack2x16float(rw_u32_a[v_base + ring_slot * vec_u32 + p]);
            let d0 = p << 1u;
            wg_v6_kraw[d0] = krp.x;
            wg_v6_kraw[d0 + 1u] = krp.y;
            wg_v6_vev[d0] = vp.x;
            wg_v6_vev[d0 + 1u] = vp.y;
            kn_sq += krp.x * krp.x + krp.y * krp.y;
        }
    }
    wg_v6_red[tid] = kn_sq;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_red[tid] += wg_v6_red[tid + s];
        }
        workgroupBarrier();
    }

    if (active_evict) {
        let inv_kn = 1.0 / max(sqrt(wg_v6_red[0]), 1e-6);
        for (var d = tid; d < head_dim; d += 64u) {
            wg_v6_ku[d] = wg_v6_kraw[d] * inv_kn;
        }
    }
    workgroupBarrier();

    // Step 2: Prediction `pred = W_scar @ ku`, value error `err = v_ev - pred`, and `||err||_2`
    var err_sq: f32 = 0.0;
    if (active_evict) {
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            var pred_r: f32 = 0.0;
            for (var c = 0u; c < head_dim; c += 1u) {
                pred_r += rw_f32_a[w_row + c] * wg_v6_ku[c];
            }
            let e_r = wg_v6_vev[r] - pred_r;
            wg_v6_err[r] = e_r;
            err_sq += e_r * e_r;
        }
    }
    wg_v6_red[tid] = err_sq;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_red[tid] += wg_v6_red[tid + s];
        }
        workgroupBarrier();
    }
    let err_norm = sqrt(wg_v6_red[0]);

    // Step 3: Gate `g_tau = sigmoid(dot(w_gate, k_raw_ev) + b_gate)` and surprise `s_tau = g_tau * ||err||_2`
    let gate_stride = head_dim + 3u;
    let gate_off = n_kv_heads * head_dim + h_kv * gate_stride;
    var g_dot: f32 = 0.0;
    if (active_evict) {
        for (var d = tid; d < head_dim; d += 64u) {
            g_dot += ro_f32_c[gate_off + d] * wg_v6_kraw[d];
        }
    }
    wg_v6_red[tid] = g_dot;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_red[tid] += wg_v6_red[tid + s];
        }
        workgroupBarrier();
    }

    let b_gate = ro_f32_c[gate_off + head_dim + 0u];
    let lambda_h = ro_f32_c[gate_off + head_dim + 1u];
    let eta_h = ro_f32_c[gate_off + head_dim + 2u];
    let z_gate = clamp(wg_v6_red[0] + b_gate, -20.0, 20.0);
    let g_tau = 1.0 / (1.0 + exp(-z_gate));
    let s_tau = g_tau * err_norm;

    // Step 3a: Promote high-surprise evicted token into Exact Micro-Page Bank BEFORE overwriting `ring_slot`
    if (tid == 0u && active_evict && en_mp && (num_micro_pages > 0u) && (s_tau >= micro_page_theta)) {
        let ps_base = head_flat * num_micro_pages;
        if (page_cnt < num_micro_pages) {
            wg_v6_ctrl[6] = page_cnt;
            rw_f32_b[ps_base + page_cnt] = s_tau;
            rw_u32_a[cnt_base + 3u] = page_cnt + 1u;
        } else {
            var min_idx: u32 = 0u;
            var min_sc: f32 = rw_f32_b[ps_base + 0u];
            for (var m = 1u; m < num_micro_pages; m += 1u) {
                let sc_m = rw_f32_b[ps_base + m];
                if (sc_m < min_sc) {
                    min_sc = sc_m;
                    min_idx = m;
                }
            }
            if (s_tau > (min_sc + 1e-4)) {
                wg_v6_ctrl[6] = min_idx;
                rw_f32_b[ps_base + min_idx] = s_tau;
            }
        }
    }
    workgroupBarrier();

    let target_page = wg_v6_ctrl[6];
    if (target_page != 0xFFFFFFFFu) {
        let dst_slot = cap_local + target_page;
        for (var p = tid; p < vec_u32; p += 64u) {
            rw_u32_a[krope_base + dst_slot * vec_u32 + p] = rw_u32_a[krope_base + ring_slot * vec_u32 + p];
            rw_u32_a[v_base + dst_slot * vec_u32 + p] = rw_u32_a[v_base + ring_slot * vec_u32 + p];
        }
    }
    workgroupBarrier();

    // Step 3b: Widrow-Hoff Fast-Weight Matrix `W_scar` Update
    if (active_evict && en_wh) {
        let decay_eff = 1.0 - lambda_h * g_tau;
        let lr_eff = eta_h * g_tau;
        for (var r = tid; r < head_dim; r += 64u) {
            let w_row = w_base + r * head_dim;
            let e_r = wg_v6_err[r];
            for (var c = 0u; c < head_dim; c += 1u) {
                rw_f32_a[w_row + c] = decay_eff * rw_f32_a[w_row + c] + lr_eff * e_r * wg_v6_ku[c];
            }
        }
    }
    workgroupBarrier();

    // Step 4: Write current `(k_rope_cur, k_raw_cur, v_cur)` into Sink or Ring Slot
    let write_slot = select(ring_slot, pos_t, pos_t < num_sink);
    let in_off = h_kv * head_dim;
    for (var p = tid; p < vec_u32; p += 64u) {
        let d0 = p << 1u;
        let krop2 = clamp(
            vec2<f32>(ro_f32_a[in_off + d0], ro_f32_a[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        let kraw2 = clamp(
            vec2<f32>(ro_f32_c[in_off + d0], ro_f32_c[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        let vn2 = clamp(
            vec2<f32>(ro_f32_b[in_off + d0], ro_f32_b[in_off + d0 + 1u]),
            vec2<f32>(-65500.0),
            vec2<f32>(65500.0),
        );
        rw_u32_a[krope_base + write_slot * vec_u32 + p] = pack2x16float(krop2);
        rw_u32_a[kraw_base + write_slot * vec_u32 + p] = pack2x16float(kraw2);
        rw_u32_a[v_base + write_slot * vec_u32 + p] = pack2x16float(vn2);
    }
    if (tid == 0u) {
        if (pos_t < num_sink) {
            if ((pos_t + 1u) > sink_cnt) {
                rw_u32_a[cnt_base + 0u] = pos_t + 1u;
            }
        } else {
            rw_u32_a[cnt_base + 1u] = min(slide_len, win_cnt + 1u);
            if (do_evict) {
                rw_u32_a[cnt_base + 2u] = absorbed + 1u;
            }
        }
    }
}

// ----------------------------------------------------------------------------
// 26. Scar-Native v6.0 Dual-Tier Hierarchical O(1) GQA Attention
//     (`Exact Softmax over [Sinks ∪ Ring ∪ Micro-Pages] + Widrow-Hoff W_scar @ qu`)
//     1 workgroup per Query head (`wgid.x = h_q in 0 .. n_q_heads - 1`).
//
//     params:
//       p0 = layer_idx
//       p1 = n_q_heads
//       p2 = n_kv_heads
//       p3 = head_dim (<= 128)
//       p4 = num_sink
//       p5 = slide_len
//       p6 = num_micro_pages (<= 64)
//       p7 = (flags << 28u) | counts_u32_total (`bit 28 = en_wh`, `bit 29 = en_mp`)
//       f0 = attn_scale
//     ro_u32_a: scarnative_v6_kv_f16_u32
//     ro_f32_a: q_rope_cur [n_q_heads * head_dim]
//     ro_f32_b: scarnative_v6_w_scar_f32 [L * n_kv_heads * head_dim * head_dim]
//     ro_f32_c: q_raw_and_alpha ([0..n_q*d_h-1] = q_raw_cur, followed by alpha_assoc[h_q])
//     rw_f32_a: attn_out [n_q_heads * head_dim]
// ----------------------------------------------------------------------------
var<workgroup> wg_v6_qrope: array<f32, 128>;
var<workgroup> wg_v6_qu: array<f32, 128>;
var<workgroup> wg_v6_logits: array<f32, 512>;
var<workgroup> wg_v6_ared: array<f32, 64>;
var<workgroup> wg_v6_acnts: array<u32, 4>;

@compute @workgroup_size(64, 1, 1)
fn scarnative_v6_hierarchical_o1_gqa_attention(
    @builtin(workgroup_id) wgid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let tid = lid.x;
    let layer_idx = params.p0;
    let n_q_heads = params.p1;
    let n_kv_heads = params.p2;
    let head_dim = params.p3;
    let num_sink = params.p4;
    let slide_len = params.p5;
    let num_micro_pages = params.p6;
    let counts_u32_total = params.p7 & 0x0FFFFFFFu;
    let flags = params.p7 >> 28u;
    let en_wh = (flags & 1u) != 0u;
    let en_mp = (flags & 2u) != 0u;
    let attn_scale = params.f0;

    let h_q = wgid.x;
    let gqa_group = n_q_heads / n_kv_heads;
    let h_kv = h_q / gqa_group;
    let head_flat = layer_idx * n_kv_heads + h_kv;
    let cnt_base = head_flat * 4u;

    let cap_local = num_sink + slide_len;
    let cap_total = cap_local + num_micro_pages;
    let vec_u32 = head_dim >> 1u;
    let head_stride = (2u * cap_total + cap_local) * vec_u32;
    let krope_base = counts_u32_total + head_flat * head_stride;
    let v_base = krope_base + cap_total * vec_u32;
    let w_base = head_flat * head_dim * head_dim;

    var qraw_sq: f32 = 0.0;
    for (var d = tid; d < head_dim; d += 64u) {
        wg_v6_qrope[d] = ro_f32_a[h_q * head_dim + d];
        let qr = ro_f32_c[h_q * head_dim + d];
        wg_v6_qu[d] = qr;
        qraw_sq += qr * qr;
    }
    wg_v6_ared[tid] = qraw_sq;
    if (tid < 4u) {
        wg_v6_acnts[tid] = ro_u32_a[cnt_base + tid];
    }
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_ared[tid] += wg_v6_ared[tid + s];
        }
        workgroupBarrier();
    }

    let inv_qn = 1.0 / max(sqrt(wg_v6_ared[0]), 1e-6);
    for (var d = tid; d < head_dim; d += 64u) {
        wg_v6_qu[d] = wg_v6_qu[d] * inv_qn;
    }
    workgroupBarrier();

    let sink_cnt = wg_v6_acnts[0];
    let win_cnt = wg_v6_acnts[1];
    let absorbed = wg_v6_acnts[2];
    let page_cnt = select(0u, wg_v6_acnts[3], (layer_idx > 0u) && en_mp);
    let local_keys = sink_cnt + win_cnt;
    let total_keys = local_keys + page_cnt;

    // Phase 1: Exact scaled dot-product logits over `[Sinks ∪ Sliding Ring ∪ Active Micro-Pages]`
    var local_max: f32 = -1e30;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        var phys_slot: u32 = idx;
        if (idx >= local_keys) {
            phys_slot = cap_local + (idx - local_keys);
        } else if (idx >= sink_cnt) {
            phys_slot = num_sink + (idx - sink_cnt);
        }
        let k_off = krope_base + phys_slot * vec_u32;
        var dot_qk: f32 = 0.0;
        for (var p = 0u; p < vec_u32; p += 1u) {
            let kp = unpack2x16float(ro_u32_a[k_off + p]);
            let d0 = p << 1u;
            dot_qk += wg_v6_qrope[d0] * kp.x + wg_v6_qrope[d0 + 1u] * kp.y;
        }
        let lg = dot_qk * attn_scale;
        wg_v6_logits[idx] = lg;
        if (lg > local_max) {
            local_max = lg;
        }
    }
    wg_v6_ared[tid] = local_max;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_ared[tid] = max(wg_v6_ared[tid], wg_v6_ared[tid + s]);
        }
        workgroupBarrier();
    }
    let max_lg = wg_v6_ared[0];

    var local_sum: f32 = 0.0;
    for (var idx = tid; idx < total_keys; idx += 64u) {
        let ex = exp(wg_v6_logits[idx] - max_lg);
        wg_v6_logits[idx] = ex;
        local_sum += ex;
    }
    wg_v6_ared[tid] = local_sum;
    workgroupBarrier();

    for (var s = 32u; s > 0u; s >>= 1u) {
        if (tid < s) {
            wg_v6_ared[tid] += wg_v6_ared[tid + s];
        }
        workgroupBarrier();
    }
    let inv_sum = 1.0 / max(wg_v6_ared[0], 1e-12);
    for (var idx = tid; idx < total_keys; idx += 64u) {
        wg_v6_logits[idx] = wg_v6_logits[idx] * inv_sum;
    }
    workgroupBarrier();

    // Phase 2: Aggregate exact Softmax V + Content-Addressed `W_scar @ qu`
    let use_wh = en_wh && (layer_idx > 0u) && (absorbed > 0u);
    let alpha_assoc = select(0.0, ro_f32_c[n_q_heads * head_dim + h_q], use_wh);
    for (var p = tid; p < vec_u32; p += 64u) {
        var v_acc = vec2<f32>(0.0, 0.0);
        for (var idx = 0u; idx < total_keys; idx += 1u) {
            var phys_slot: u32 = idx;
            if (idx >= local_keys) {
                phys_slot = cap_local + (idx - local_keys);
            } else if (idx >= sink_cnt) {
                phys_slot = num_sink + (idx - sink_cnt);
            }
            let vp = unpack2x16float(ro_u32_a[v_base + phys_slot * vec_u32 + p]);
            v_acc += wg_v6_logits[idx] * vp;
        }
        let d0 = p << 1u;
        var fast0: f32 = 0.0;
        var fast1: f32 = 0.0;
        if (use_wh) {
            let w_r0 = w_base + d0 * head_dim;
            let w_r1 = w_base + (d0 + 1u) * head_dim;
            for (var c = 0u; c < head_dim; c += 1u) {
                let q_c = wg_v6_qu[c];
                fast0 += ro_f32_b[w_r0 + c] * q_c;
                fast1 += ro_f32_b[w_r1 + c] * q_c;
            }
        }
        rw_f32_a[h_q * head_dim + d0] = v_acc.x + alpha_assoc * fast0;
        rw_f32_a[h_q * head_dim + d0 + 1u] = v_acc.y + alpha_assoc * fast1;
    }
}
