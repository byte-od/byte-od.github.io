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
            let kev2 = vec2<f32>(wg_sc_kev[d0], wg_sc_kev[d0 + 1u]);
            let vev2 = vec2<f32>(wg_sc_vev[d0], wg_sc_vev[d0 + 1u]);
            if (slot_action == 1u) {
                rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(kev2);
                rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(vev2);
            } else {
                let old_k = unpack2x16float(rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p]);
                let old_v = unpack2x16float(rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p]);
                rw_u32_a[k_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(0.65 * old_k + 0.35 * kev2);
                rw_u32_a[v_base_u32 + dst_slot * vec_u32 + p] = pack2x16float(0.22 * old_v + 0.78 * vev2);
            }
        }
    }
    workgroupBarrier();

    // Step 4: Write current token `(k_cur, v_cur)` into Sink or Fixed Ring Slot
    let write_slot = select(evict_slot, pos_t, pos_t < num_sink);
    let in_head_off = h_kv * head_dim;
    for (var p = tid; p < vec_u32; p += 64u) {
        let d0 = p << 1u;
        let kn2 = vec2<f32>(ro_f32_a[in_head_off + d0], ro_f32_a[in_head_off + d0 + 1u]);
        let vn2 = vec2<f32>(ro_f32_b[in_head_off + d0], ro_f32_b[in_head_off + d0 + 1u]);
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
        let x = ro_f32_a[idx];
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
