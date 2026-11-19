// ============================================================================
// ScarField v9.2-Elastic — Pure Post-Transformer WebGPU WGSL 1.0 Compute Kernels
// ============================================================================
// STRICT ARCHITECTURAL INTEGRITY (`AGENTS.md` Zero-Facade Guarantee):
//   - ZERO dense FP16/FP32 weight decompression in VRAM: all `E8SphericalLinear`
//     projections execute directly from packed `1.000 bpw` Gosset `E_8` `u32` codes
//     + `2-bit` ternary `{-1, 0, +1}` residual scars + shared `256 x 8` `E_8` L1 LUT!
//   - ZERO `RMSNorm` / `LayerNorm`: exact `L2` hyperspherical projection `||h||_2 == 1`.
//   - ZERO `ReLU` / `GELU` / `SwiGLU`: `C^1` spherical-cap horizon culling +
//     rank-`2r` Symplectic Cayley-Woodbury orthogonal steps on `SO(D)`.
//   - ZERO `O(T)` KV Cache: strictly `O(1)` Kalman-Scar `W_fast` + `16 x 32 = 512`
//     Multi-Bucket Biorthogonal Shock Bank (`K_tilde = (K K^T + lam I)^(-1) K`).
//   - ZERO `Softmax` / `exp`: exact Tsallis `alpha = 1.5` spherical-cap simplex
//     (`> 95%` exact zeros + native `0x00` physical vacuum abstention).
// ============================================================================

struct FwhtParams {
  dim: u32,
  inverse_flag: u32,
  _pad0: u32,
  _pad1: u32,
};

@group(0) @binding(0) var<uniform> fwht_cfg: FwhtParams;
@group(0) @binding(1) var<storage, read> fwht_in: array<f32>;
@group(0) @binding(2) var<storage, read> fwht_signs: array<f32>;
@group(0) @binding(3) var<storage, read_write> fwht_out: array<f32>;

var<workgroup> sh_fwht: array<f32, 64>;

// Kernel #1: Multiplierless 64-point Fast Walsh-Hadamard + Deterministic Rademacher Rotation
@compute @workgroup_size(64)
fn sf_fwht64_rademacher(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let blk = wid.x;
  let tid = lid.x;
  if (blk * 64u >= fwht_cfg.dim) {
    return;
  }
  let idx = blk * 64u + tid;
  var sgn: f32 = 1.0;
  var v: f32 = 0.0;
  if (idx < fwht_cfg.dim) {
    sgn = fwht_signs[idx];
    v = fwht_in[idx];
    if (fwht_cfg.inverse_flag == 0u) {
      v = v * sgn;
    }
  }
  sh_fwht[tid] = v;
  workgroupBarrier();

  // 6 butterfly stages (`stride = 1, 2, 4, 8, 16, 32`)
  for (var st: u32 = 0u; st < 6u; st = st + 1u) {
    let stride = 1u << st;
    let partner = tid ^ stride;
    let a = sh_fwht[tid];
    let b = sh_fwht[partner];
    workgroupBarrier();
    if ((tid & stride) == 0u) {
      sh_fwht[tid] = a + b;
    } else {
      sh_fwht[tid] = b - a;
    }
    workgroupBarrier();
  }

  var res = sh_fwht[tid] * 0.125;
  if (fwht_cfg.inverse_flag != 0u) {
    res = res * sgn;
  }
  if (idx < fwht_cfg.dim) {
    fwht_out[idx] = res;
  }
}

// ----------------------------------------------------------------------------
// Kernel #2a & #2b: `1.000 bpw` Gosset `E_8` + `2-bit` Ternary Scar GEMV
// ----------------------------------------------------------------------------
struct E8GemvParams {
  out_features: u32,
  in_features: u32,
  scar_rank: u32,
  _pad0: u32,
  scale_base: f32,  // tau_base / sqrt(in_features)
  scale_scar: f32,  // gamma_scar / sqrt(in_features * max(1, scar_rank))
  _pad1: f32,
  _pad2: f32,
};

@group(0) @binding(0) var<uniform> e8_cfg: E8GemvParams;
@group(0) @binding(1) var<storage, read> e8_x_raw: array<f32>;
@group(0) @binding(2) var<storage, read> e8_a_tern_u32: array<u32>;
@group(0) @binding(3) var<storage, read_write> e8_xa_out: array<f32>;

var<workgroup> sh_red64: array<f32, 64>;

fn unpack_trit_from_u32(word: u32, sub_idx: u32) -> f32 {
  let shift = sub_idx * 2u;
  let code = (word >> shift) & 3u;
  return f32(i32(code) - 1);
}

@compute @workgroup_size(64)
fn sf_ternary_a_gemv(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let k = wid.x;
  let tid = lid.x;
  if (k >= e8_cfg.scar_rank) {
    return;
  }
  let words_per_row = e8_cfg.in_features / 16u;
  let row_base = k * words_per_row;
  var acc: f32 = 0.0;
  for (var w_i: u32 = tid; w_i < words_per_row; w_i = w_i + 64u) {
    let word = e8_a_tern_u32[row_base + w_i];
    let x_base = w_i * 16u;
    for (var j: u32 = 0u; j < 16u; j = j + 1u) {
      acc = acc + unpack_trit_from_u32(word, j) * e8_x_raw[x_base + j];
    }
  }
  sh_red64[tid] = acc;
  workgroupBarrier();
  for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sh_red64[tid] = sh_red64[tid] + sh_red64[tid + s];
    }
    workgroupBarrier();
  }
  if (tid == 0u) {
    e8_xa_out[k] = sh_red64[0];
  }
}

@group(0) @binding(0) var<uniform> e8m_cfg: E8GemvParams;
@group(0) @binding(1) var<storage, read> e8m_x_rot: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> e8m_lut: array<vec4<f32>>;      // 512 x vec4<f32> = 8 KB (`[256, 8]`)
@group(0) @binding(3) var<storage, read> e8m_codes_u32: array<u32>;     // [O, I / 32] (`4` E8 bytes = 32 weights per u32!)
@group(0) @binding(4) var<storage, read> e8m_b_tern_u32: array<u32>;    // packed 2-bit `[O * r / 16]`
@group(0) @binding(5) var<storage, read> e8m_xa: array<f32>;            // [r]
@group(0) @binding(6) var<storage, read_write> e8m_y: array<f32>;       // [O]

var<workgroup> sh_e8_lut: array<vec4<f32>, 512>;
var<workgroup> sh_xa_vec: array<f32, 32>;

@compute @workgroup_size(64)
fn sf_e8_ternary_gemv(
  @builtin(global_invocation_id) gid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let tid = lid.x;
  // Cooperative load of shared 8 KB `[256, 8]` Gosset E8 codebook into GPU L1 SRAM
  for (var i: u32 = tid; i < 512u; i = i + 64u) {
    sh_e8_lut[i] = e8m_lut[i];
  }
  if (tid < e8m_cfg.scar_rank && tid < 32u) {
    sh_xa_vec[tid] = e8m_xa[tid];
  }
  workgroupBarrier();

  let row = gid.x;
  if (row >= e8m_cfg.out_features) {
    return;
  }

  let u32_per_row = e8m_cfg.in_features / 32u;
  let c_base = row * u32_per_row;
  var dot_e8: f32 = 0.0;

  for (var u_i: u32 = 0u; u_i < u32_per_row; u_i = u_i + 1u) {
    let packed4 = e8m_codes_u32[c_base + u_i];
    let x_vec_base = u_i * 8u; // 4 blocks of 8 floats = 8 `vec4<f32>`

    let c0 = (packed4 & 0xFFu) * 2u;
    let c1 = ((packed4 >> 8u) & 0xFFu) * 2u;
    let c2 = ((packed4 >> 16u) & 0xFFu) * 2u;
    let c3 = ((packed4 >> 24u) & 0xFFu) * 2u;

    dot_e8 = dot_e8
      + dot(sh_e8_lut[c0], e8m_x_rot[x_vec_base])
      + dot(sh_e8_lut[c0 + 1u], e8m_x_rot[x_vec_base + 1u])
      + dot(sh_e8_lut[c1], e8m_x_rot[x_vec_base + 2u])
      + dot(sh_e8_lut[c1 + 1u], e8m_x_rot[x_vec_base + 3u])
      + dot(sh_e8_lut[c2], e8m_x_rot[x_vec_base + 4u])
      + dot(sh_e8_lut[c2 + 1u], e8m_x_rot[x_vec_base + 5u])
      + dot(sh_e8_lut[c3], e8m_x_rot[x_vec_base + 6u])
      + dot(sh_e8_lut[c3 + 1u], e8m_x_rot[x_vec_base + 7u]);
  }

  var y_val = dot_e8 * e8m_cfg.scale_base;

  if (e8m_cfg.scar_rank > 0u && abs(e8m_cfg.scale_scar) > 0.0) {
    var dot_s: f32 = 0.0;
    let r = e8m_cfg.scar_rank;
    for (var k: u32 = 0u; k < r; k = k + 1u) {
      let flat_trit_idx = row * r + k;
      let word_b = e8m_b_tern_u32[flat_trit_idx >> 4u];
      let sub_b = flat_trit_idx & 15u;
      dot_s = dot_s + unpack_trit_from_u32(word_b, sub_b) * sh_xa_vec[k];
    }
    y_val = y_val + dot_s * e8m_cfg.scale_scar;
  }

  e8m_y[row] = y_val;
}

// ----------------------------------------------------------------------------
// Kernel #3: Exact Unit-Sphere Projection `S^{D-1}` (Zero `RMSNorm`!)
// ----------------------------------------------------------------------------
struct SphereProjParams {
  dim: u32,
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> sp_cfg: SphereProjParams;
@group(0) @binding(1) var<storage, read> sp_in: array<f32>;
@group(0) @binding(2) var<storage, read_write> sp_out: array<f32>;

var<workgroup> sh_sp: array<f32, 256>;

@compute @workgroup_size(256)
fn sf_unit_sphere_project(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  var sq: f32 = 0.0;
  for (var i: u32 = tid; i < sp_cfg.dim; i = i + 256u) {
    let v = sp_in[i];
    sq = sq + v * v;
  }
  sh_sp[tid] = sq;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sh_sp[tid] = sh_sp[tid] + sh_sp[tid + s];
    }
    workgroupBarrier();
  }
  let inv_nrm = 1.0 / max(1e-8, sqrt(sh_sp[0]));
  for (var i: u32 = tid; i < sp_cfg.dim; i = i + 256u) {
    sp_out[i] = sp_in[i] * inv_nrm;
  }
}

// ----------------------------------------------------------------------------
// Kernel #4: Rank-`2r` Symplectic Cayley-Woodbury Geodesic Step on `SO(D)`
// ----------------------------------------------------------------------------
struct CayleyParams {
  dim: u32,
  rank_r: u32, // r <= 20 -> 2r <= 40
  dt: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> cay_cfg: CayleyParams;
@group(0) @binding(1) var<storage, read> cay_x: array<f32>;         // [D]
@group(0) @binding(2) var<storage, read> cay_uv: array<f32>;        // [D, 2r] row-major
@group(0) @binding(3) var<storage, read_write> cay_out: array<f32>; // [D]

// Max `2r = 40`: `40 x 40 = 1600` floats (`6.4 KB` SRAM) + vectors -> `< 8 KB` workgroup SRAM!
var<workgroup> sh_m_mat: array<f32, 1600>;
var<workgroup> sh_rhs: array<f32, 40>;
var<workgroup> sh_wp: array<f32, 40>;
var<workgroup> sh_dot64: array<f32, 64>;

@compute @workgroup_size(64)
fn sf_cayley_woodbury_step(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let d = cay_cfg.dim;
  let r = cay_cfg.rank_r;
  let two_r = 2u * r;
  let alpha = 0.5 * cay_cfg.dt;

  // 1. Compute `uvt_x[m] = sum_i uv[i, m] * x[i]` for `m in [0 .. 2r - 1]`
  for (var m: u32 = 0u; m < two_r; m = m + 1u) {
    var s_x: f32 = 0.0;
    for (var i: u32 = tid; i < d; i = i + 64u) {
      s_x = s_x + cay_uv[i * two_r + m] * cay_x[i];
    }
    sh_dot64[tid] = s_x;
    workgroupBarrier();
    for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
      if (tid < s) {
        sh_dot64[tid] = sh_dot64[tid] + sh_dot64[tid + s];
      }
      workgroupBarrier();
    }
    if (tid == 0u) {
      // qt_x = [uvt_x[r:], uvt_x[:r]]
      let dst_m = select(m - r, m + r, m < r);
      sh_rhs[dst_m] = sh_dot64[0];
    }
    workgroupBarrier();
  }

  // 2. Compute symmetric `g_gram[a, b] = sum_i uv[i, a] * uv[i, b]` and map directly into `M = I + alpha * qt_p`
  for (var a: u32 = 0u; a < two_r; a = a + 1u) {
    for (var b: u32 = a; b < two_r; b = b + 1u) {
      var s_g: f32 = 0.0;
      for (var i: u32 = tid; i < d; i = i + 64u) {
        s_g = s_g + cay_uv[i * two_r + a] * cay_uv[i * two_r + b];
      }
      sh_dot64[tid] = s_g;
      workgroupBarrier();
      for (var s: u32 = 32u; s > 0u; s = s >> 1u) {
        if (tid < s) {
          sh_dot64[tid] = sh_dot64[tid] + sh_dot64[tid + s];
        }
        workgroupBarrier();
      }
      if (tid == 0u) {
        let g_val = sh_dot64[0];
        // Write both (a, b) and (b, a) into `qt_p`:
        // g_neg[row, col] = +g[row, col] if col < r else -g[row, col]
        // qt_p[dst_r, col] = g_neg[(dst_r + r) % 2r, col]
        let row_ab = select(a - r, a + r, a < r);
        let sgn_b = select(-1.0, 1.0, b < r);
        sh_m_mat[row_ab * two_r + b] = select(0.0, 1.0, row_ab == b) + alpha * (sgn_b * g_val);

        if (b != a) {
          let row_ba = select(b - r, b + r, b < r);
          let sgn_a = select(-1.0, 1.0, a < r);
          sh_m_mat[row_ba * two_r + a] = select(0.0, 1.0, row_ba == a) + alpha * (sgn_a * g_val);
        }
      }
      workgroupBarrier();
    }
  }

  // 3. Solve `[2r x 2r]` linear system `M w = qt_x` (`det(M) >= 1.0`) via partial-pivoted Gauss-Jordan in SRAM
  for (var k: u32 = 0u; k < two_r; k = k + 1u) {
    if (tid == 0u) {
      var piv = k;
      var max_a = abs(sh_m_mat[k * two_r + k]);
      for (var p: u32 = k + 1u; p < two_r; p = p + 1u) {
        let cand = abs(sh_m_mat[p * two_r + k]);
        if (cand > max_a) {
          max_a = cand;
          piv = p;
        }
      }
      if (piv != k) {
        for (var c: u32 = 0u; c < two_r; c = c + 1u) {
          let tmp = sh_m_mat[k * two_r + c];
          sh_m_mat[k * two_r + c] = sh_m_mat[piv * two_r + c];
          sh_m_mat[piv * two_r + c] = tmp;
        }
        let tr = sh_rhs[k];
        sh_rhs[k] = sh_rhs[piv];
        sh_rhs[piv] = tr;
      }
      let inv_diag = 1.0 / sh_m_mat[k * two_r + k];
      for (var c: u32 = k; c < two_r; c = c + 1u) {
        sh_m_mat[k * two_r + c] = sh_m_mat[k * two_r + c] * inv_diag;
      }
      sh_rhs[k] = sh_rhs[k] * inv_diag;
    }
    workgroupBarrier();

    if (tid < two_r && tid != k) {
      let factor = sh_m_mat[tid * two_r + k];
      if (factor != 0.0) {
        for (var c: u32 = k; c < two_r; c = c + 1u) {
          sh_m_mat[tid * two_r + c] = sh_m_mat[tid * two_r + c] - factor * sh_m_mat[k * two_r + c];
        }
        sh_rhs[tid] = sh_rhs[tid] - factor * sh_rhs[k];
      }
    }
    workgroupBarrier();
  }

  if (tid < two_r) {
    // w_p = [w_vec[:r], -w_vec[r:]]
    sh_wp[tid] = select(-sh_rhs[tid], sh_rhs[tid], tid < r);
  }
  workgroupBarrier();

  // 4. Reconstruct `x_next = x - dt * (uv @ w_p)` (`||x_next||_2 == 1.000000`)
  let dt_f = cay_cfg.dt;
  for (var i: u32 = tid; i < d; i = i + 64u) {
    var pw_i: f32 = 0.0;
    let base_i = i * two_r;
    for (var m: u32 = 0u; m < two_r; m = m + 1u) {
      pw_i = pw_i + cay_uv[base_i + m] * sh_wp[m];
    }
    cay_out[i] = cay_x[i] - dt_f * pw_i;
  }
}

// ----------------------------------------------------------------------------
// Kernel #5: `O(1)` Kalman-Scar `W_fast` Causal Readout & Rank-1 Innovation Update
// ----------------------------------------------------------------------------
struct KalmanFastParams {
  d_bus: u32,
  d_key: u32,
  d_sub: u32,
  mutate_flag: u32,
};

@group(0) @binding(0) var<uniform> kf_cfg: KalmanFastParams;
@group(0) @binding(1) var<storage, read_write> kf_w_fast: array<f32>; // [d_bus, d_key] (`[G, d_sub, d_key]`)
@group(0) @binding(2) var<storage, read> kf_q: array<f32>;            // [d_key]
@group(0) @binding(3) var<storage, read> kf_k_bind: array<f32>;       // [d_key]
@group(0) @binding(4) var<storage, read> kf_v: array<f32>;            // [d_bus]
@group(0) @binding(5) var<storage, read> kf_beta_bind: array<f32>;    // [G]
@group(0) @binding(6) var<storage, read_write> kf_kalman_out: array<f32>; // [d_bus]

@compute @workgroup_size(64)
fn sf_kalman_wfast_step(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  if (row >= kf_cfg.d_bus) {
    return;
  }
  let dk = kf_cfg.d_key;
  let w_base = row * dk;
  var r_q: f32 = 0.0;
  var r_k: f32 = 0.0;
  for (var j: u32 = 0u; j < dk; j = j + 1u) {
    let w_ij = kf_w_fast[w_base + j];
    r_q = r_q + w_ij * kf_q[j];
    r_k = r_k + w_ij * kf_k_bind[j];
  }
  kf_kalman_out[row] = r_q;

  if (kf_cfg.mutate_flag != 0u) {
    let g_idx = row / kf_cfg.d_sub;
    let delta_v = kf_beta_bind[g_idx] * (kf_v[row] - r_k);
    if (delta_v != 0.0) {
      for (var j: u32 = 0u; j < dk; j = j + 1u) {
        kf_w_fast[w_base + j] = kf_w_fast[w_base + j] + delta_v * kf_k_bind[j];
      }
    }
  }
}

// ----------------------------------------------------------------------------
// Kernel #6a & #6b: Multi-Bucket `16 x 32 = 512` Biorthogonal Dual Shock Readout
// ----------------------------------------------------------------------------
struct BiorthParams {
  num_buckets: u32,
  m_sub: u32,       // 32 slots per bucket
  d_key: u32,
  d_bus: u32,
  kappa: f32,
  ridge_lambda: f32,
  biorth_gain: f32,
  _pad: f32,
};

@group(0) @binding(0) var<uniform> bio_cfg: BiorthParams;
@group(0) @binding(1) var<storage, read> bio_q: array<f32>;             // [d_key]
@group(0) @binding(2) var<storage, read> bio_keys: array<f32>;          // [M, d_key]
@group(0) @binding(3) var<storage, read> bio_valid: array<u32>;         // [M] (0 or 1)
@group(0) @binding(4) var<storage, read_write> bio_alpha: array<f32>;   // [M]

var<workgroup> sh_gram32: array<f32, 1024>; // 32 x 32 SPD Gram per bucket
var<workgroup> sh_cos32: array<f32, 32>;
var<workgroup> sh_val32: array<u32, 32>;
var<workgroup> sh_hor_gate: f32;

fn c1_cap_scalar(cos_sim: f32, kappa: f32) -> f32 {
  let delta = cos_sim - kappa;
  if (delta <= 0.0) {
    return 0.0;
  }
  let w = max(0.08, 1.0 - kappa);
  let d2 = delta * delta;
  return d2 / (w * w + d2);
}

@compute @workgroup_size(32)
fn sf_biorth_multibucket_alpha(
  @builtin(workgroup_id) wid: vec3<u32>,
  @builtin(local_invocation_id) lid: vec3<u32>,
) {
  let b_idx = wid.x;
  let tid = lid.x;
  let m_sub = bio_cfg.m_sub;
  if (b_idx >= bio_cfg.num_buckets) {
    return;
  }

  let slot = b_idx * m_sub + tid;
  var is_val: u32 = 0u;
  if (tid < m_sub) {
    is_val = bio_valid[slot];
  }
  sh_val32[tid] = is_val;

  var raw_c: f32 = 0.0;
  let dk = bio_cfg.d_key;
  if (tid < m_sub && is_val != 0u) {
    let k_base = slot * dk;
    for (var j: u32 = 0u; j < dk; j = j + 1u) {
      raw_c = raw_c + bio_q[j] * bio_keys[k_base + j];
    }
  }
  sh_cos32[tid] = raw_c;
  workgroupBarrier();

  if (tid == 0u) {
    var max_c: f32 = -1.0;
    for (var s: u32 = 0u; s < m_sub; s = s + 1u) {
      if (sh_val32[s] != 0u) {
        max_c = max(max_c, sh_cos32[s]);
      }
    }
    let kap = clamp(bio_cfg.kappa, 0.15, 0.88);
    sh_hor_gate = clamp(2.0 * c1_cap_scalar(max_c, kap), 0.0, 1.0);
  }
  let hor_gate = workgroupUniformLoad(&sh_hor_gate);
  if (hor_gate <= 0.0) {
    if (tid < m_sub) {
      bio_alpha[slot] = 0.0;
    }
    return;
  }

  // Build `[32 x 32]` SPD Gram `G[tid, col] = <K_tid, K_col> + ridge_lambda * delta`
  if (tid < m_sub) {
    for (var col: u32 = 0u; col < m_sub; col = col + 1u) {
      var g_tc: f32 = 0.0;
      if (is_val != 0u && sh_val32[col] != 0u) {
        let base_t = slot * dk;
        let base_c = (b_idx * m_sub + col) * dk;
        for (var j: u32 = 0u; j < dk; j = j + 1u) {
          g_tc = g_tc + bio_keys[base_t + j] * bio_keys[base_c + j];
        }
      }
      if (tid == col) {
        g_tc = g_tc + bio_cfg.ridge_lambda;
      }
      sh_gram32[tid * m_sub + col] = g_tc;
    }
  }
  workgroupBarrier();

  // Solve SPD `[32 x 32]` `G * alpha = raw_cos_valid` in shared memory
  for (var k: u32 = 0u; k < m_sub; k = k + 1u) {
    if (tid == 0u) {
      let inv_d = 1.0 / sh_gram32[k * m_sub + k];
      for (var c: u32 = k; c < m_sub; c = c + 1u) {
        sh_gram32[k * m_sub + c] = sh_gram32[k * m_sub + c] * inv_d;
      }
      sh_cos32[k] = sh_cos32[k] * inv_d;
    }
    workgroupBarrier();
    if (tid < m_sub && tid != k) {
      let f = sh_gram32[tid * m_sub + k];
      if (f != 0.0) {
        for (var c: u32 = k; c < m_sub; c = c + 1u) {
          sh_gram32[tid * m_sub + c] = sh_gram32[tid * m_sub + c] - f * sh_gram32[k * m_sub + c];
        }
        sh_cos32[tid] = sh_cos32[tid] - f * sh_cos32[k];
      }
    }
    workgroupBarrier();
  }

  if (tid < m_sub) {
    let a_val = select(0.0, hor_gate * clamp(sh_cos32[tid], 0.0, 1.5), is_val != 0u);
    bio_alpha[slot] = a_val;
  }
}

@group(0) @binding(0) var<uniform> biov_cfg: BiorthParams;
@group(0) @binding(1) var<storage, read> biov_alpha: array<f32>;        // [M]
@group(0) @binding(2) var<storage, read> biov_vals: array<f32>;         // [M, d_bus]
@group(0) @binding(3) var<storage, read_write> biov_out: array<f32>;    // [d_bus]

@compute @workgroup_size(64)
fn sf_biorth_value_combine(@builtin(global_invocation_id) gid: vec3<u32>) {
  let d_idx = gid.x;
  if (d_idx >= biov_cfg.d_bus) {
    return;
  }
  let m_tot = biov_cfg.num_buckets * biov_cfg.m_sub;
  let db = biov_cfg.d_bus;
  var acc: f32 = 0.0;
  for (var s: u32 = 0u; s < m_tot; s = s + 1u) {
    let a = biov_alpha[s];
    if (a > 0.0) {
      acc = acc + a * biov_vals[s * db + d_idx];
    }
  }
  biov_out[d_idx] = max(0.0, biov_cfg.biorth_gain) * acc;
}

// ----------------------------------------------------------------------------
// Kernel #7: Tripartite Spherical-Cap Scores + Exact `Tsallis-1.5` Simplex (`0` Softmax!)
// ----------------------------------------------------------------------------
struct HeadSimplexParams {
  d_form: u32,
  d_world: u32,
  d_bus: u32,
  repulsive_len: u32,
  mw_form: f32,
  mw_world: f32,
  mw_bus: f32,
  temp_scale: f32,
  cap_adapt_gain: f32,
  apply_repulsive: u32,
  damp_byte_a: i32,
  damp_pen_a: f32,
};

@group(0) @binding(0) var<uniform> hd_cfg: HeadSimplexParams;
@group(0) @binding(1) var<storage, read> hd_h_form: array<f32>;         // [d_form]
@group(0) @binding(2) var<storage, read> hd_h_world: array<f32>;        // [d_world]
@group(0) @binding(3) var<storage, read> hd_bus_ctx: array<f32>;        // [d_bus]
@group(0) @binding(4) var<storage, read> hd_form_anch: array<f32>;      // [256, d_form] unit-norm rows
@group(0) @binding(5) var<storage, read> hd_world_anch: array<f32>;     // [256, d_world] unit-norm rows
@group(0) @binding(6) var<storage, read> hd_bus_anch: array<f32>;       // [256, d_bus] unit-norm rows
@group(0) @binding(7) var<storage, read> hd_rep_ring: array<f32>;       // [R_rep, d_world]
@group(0) @binding(8) var<storage, read_write> hd_out: array<f32>;      // [516]: [scores(256), probs(256), meta(4)]

var<workgroup> sh_comb: array<f32, 256>;
var<workgroup> sh_wcos: array<f32, 256>;
var<workgroup> sh_scores: array<f32, 256>;
var<workgroup> sh_eff_temp: f32;
var<workgroup> sh_max_wcos: f32;
var<workgroup> sh_lambda: f32;

@compute @workgroup_size(256)
fn sf_head_scores_and_tsallis15(@builtin(local_invocation_id) lid: vec3<u32>) {
  let v = lid.x; // 1 thread per UTF-8 byte `v in [0..255]`
  var cf: f32 = 0.0;
  let bf = v * hd_cfg.d_form;
  for (var i: u32 = 0u; i < hd_cfg.d_form; i = i + 1u) {
    cf = cf + hd_h_form[i] * hd_form_anch[bf + i];
  }

  var cw: f32 = 0.0;
  let bw = v * hd_cfg.d_world;
  for (var i: u32 = 0u; i < hd_cfg.d_world; i = i + 1u) {
    cw = cw + hd_h_world[i] * hd_world_anch[bw + i];
  }

  var cb: f32 = 0.0;
  let bb = v * hd_cfg.d_bus;
  for (var i: u32 = 0u; i < hd_cfg.d_bus; i = i + 1u) {
    cb = cb + hd_bus_ctx[i] * hd_bus_anch[bb + i];
  }

  let comb_v = hd_cfg.mw_form * cf + hd_cfg.mw_world * cw + hd_cfg.mw_bus * cb;
  sh_comb[v] = comb_v;
  sh_wcos[v] = cw;
  workgroupBarrier();

  if (v == 0u) {
    // Find max_world_cos and Top-8 of `sh_comb` for exact adaptive cap temperature
    var mw_c: f32 = -2.0;
    var top8: array<f32, 8>;
    for (var k: u32 = 0u; k < 8u; k = k + 1u) {
      top8[k] = -1e9;
    }
    for (var idx: u32 = 0u; idx < 256u; idx = idx + 1u) {
      mw_c = max(mw_c, sh_wcos[idx]);
      let val = sh_comb[idx];
      if (val > top8[7]) {
        top8[7] = val;
        for (var j: i32 = 6; j >= 0; j = j - 1) {
          let ju = u32(j);
          if (top8[ju + 1u] > top8[ju]) {
            let t = top8[ju];
            top8[ju] = top8[ju + 1u];
            top8[ju + 1u] = t;
          }
        }
      }
    }
    sh_max_wcos = mw_c;
    let temp = clamp(hd_cfg.temp_scale, 1.0, 28.0);
    if (hd_cfg.cap_adapt_gain >= 0.0) {
      var mean8: f32 = 0.0;
      for (var k: u32 = 0u; k < 8u; k = k + 1u) {
        mean8 = mean8 + top8[k];
      }
      mean8 = mean8 * 0.125;
      let margin8 = max(0.0, top8[0] - mean8);
      let adapt = clamp(hd_cfg.cap_adapt_gain, 0.0, 3.0) / (1.0 + 6.0 * margin8);
      sh_eff_temp = clamp(temp / (1.0 + adapt), 1.0, 28.0);
    } else {
      sh_eff_temp = temp;
    }
  }
  workgroupBarrier();

  var sc = sh_eff_temp * comb_v;

  // Optional `AURA` repulsive ring + recent-byte anti-stutter damper
  if (hd_cfg.apply_repulsive != 0u) {
    var rep_pen: f32 = 0.0;
    for (var r_i: u32 = 0u; r_i < hd_cfg.repulsive_len; r_i = r_i + 1u) {
      var r_sim: f32 = 0.0;
      let rb_off = r_i * hd_cfg.d_world;
      for (var i: u32 = 0u; i < hd_cfg.d_world; i = i + 1u) {
        r_sim = r_sim + hd_world_anch[bw + i] * hd_rep_ring[rb_off + i];
      }
      let ex = max(0.0, r_sim - 0.88);
      rep_pen = rep_pen + ex * ex;
    }
    sc = sc - 2.5 * rep_pen;
    if (hd_cfg.damp_byte_a >= 0 && u32(hd_cfg.damp_byte_a) == v) {
      sc = sc - hd_cfg.damp_pen_a;
    }
  }

  hd_out[v] = sc;
  sh_scores[v] = sc;
  workgroupBarrier();

  // Exact Tsallis `alpha = 1.5` Spherical-Cap Simplex (`sum_v [0.5 * (sc_v - lambda*)]_+^2 == 1.0`)
  if (v == 0u) {
    var z_max: f32 = sh_scores[0];
    for (var idx: u32 = 1u; idx < 256u; idx = idx + 1u) {
      z_max = max(z_max, sh_scores[idx]);
    }
    var lo: f32 = z_max - 2.0;
    // Lower bound where `0.25 * 256 * (z_max - lo)^2 >= 1` -> `z_max - 2.0` is too high if others are low;
    // exact guaranteed lower bound: at `lo = z_max - 2.0`, single max term alone gives `[0.5 * 2.0]^2 == 1.0`!
    // And at `hi = z_max`, sum is `0.0`! So `lambda*` ALWAYS lies strictly inside `[z_max - 2.0, z_max]`!
    var hi: f32 = z_max;
    for (var it: u32 = 0u; it < 30u; it = it + 1u) {
      let mid = 0.5 * (lo + hi);
      var s_prob: f32 = 0.0;
      for (var idx: u32 = 0u; idx < 256u; idx = idx + 1u) {
        let pos = max(0.0, 0.5 * (sh_scores[idx] - mid));
        s_prob = s_prob + pos * pos;
      }
      if (s_prob > 1.0) {
        lo = mid;
      } else {
        hi = mid;
      }
    }
    sh_lambda = 0.5 * (lo + hi);
  }
  workgroupBarrier();

  let pos_v = max(0.0, 0.5 * (sc - sh_lambda));
  let prob_raw = pos_v * pos_v;
  sh_comb[v] = prob_raw;
  workgroupBarrier();

  if (v == 0u) {
    var act_c: u32 = 0u;
    var best_v: u32 = 0u;
    var best_p: f32 = -1.0;
    var sum_p: f32 = 0.0;
    for (var idx: u32 = 0u; idx < 256u; idx = idx + 1u) {
      let pv = sh_comb[idx];
      if (pv > 0.0) {
        act_c = act_c + 1u;
        sum_p = sum_p + pv;
      }
      if (pv > best_p) {
        best_p = pv;
        best_v = idx;
      }
    }
    sh_eff_temp = 1.0 / max(1e-12, sum_p);
    hd_out[512u] = sh_max_wcos;
    hd_out[513u] = sh_lambda;
    hd_out[514u] = f32(act_c);
    hd_out[515u] = f32(best_v);
  }
  let inv_sum_p = workgroupUniformLoad(&sh_eff_temp);
  hd_out[256u + v] = prob_raw * inv_sum_p;
}

// ----------------------------------------------------------------------------
// Kernel #8: Stiefel Isometric `KalmanBusState` Matrix Lift (`60M -> 302M` Live Handoff)
// ----------------------------------------------------------------------------
struct StiefelLiftParams {
  d_bus_0: u32,
  d_bus_1: u32,
  d_key: u32,
  _pad: u32,
};

@group(0) @binding(0) var<uniform> sl_cfg: StiefelLiftParams;
@group(0) @binding(1) var<storage, read> sl_w0: array<f32>;        // [d_bus_0, d_key]
@group(0) @binding(2) var<storage, read> sl_w_bus: array<f32>;     // [d_bus_0, d_bus_1]
@group(0) @binding(3) var<storage, read> sl_w_key: array<f32>;     // [d_key, d_key]
@group(0) @binding(4) var<storage, read_write> sl_w1: array<f32>;  // [d_bus_1, d_key]

@compute @workgroup_size(64)
fn sf_stiefel_wfast_lift(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row1 = gid.x;
  if (row1 >= sl_cfg.d_bus_1) {
    return;
  }
  let db0 = sl_cfg.d_bus_0;
  let db1 = sl_cfg.d_bus_1;
  let dk = sl_cfg.d_key;

  // Compute `tmp[k0] = sum_{i0=0}^{db0-1} W_bus[i0, row1] * W_0[i0, k0]`
  var tmp_k: array<f32, 128>;
  for (var k0: u32 = 0u; k0 < dk; k0 = k0 + 1u) {
    var acc0: f32 = 0.0;
    for (var i0: u32 = 0u; i0 < db0; i0 = i0 + 1u) {
      acc0 = acc0 + sl_w_bus[i0 * db1 + row1] * sl_w0[i0 * dk + k0];
    }
    tmp_k[k0] = acc0;
  }

  // Multiply by `W_key`: `W_1[row1, k1] = sum_{k0} tmp[k0] * W_key[k0, k1]`
  let out_base = row1 * dk;
  for (var k1: u32 = 0u; k1 < dk; k1 = k1 + 1u) {
    var acc1: f32 = 0.0;
    for (var k0: u32 = 0u; k0 < dk; k0 = k0 + 1u) {
      acc1 = acc1 + tmp_k[k0] * sl_w_key[k0 * dk + k1];
    }
    sl_w1[out_base + k1] = acc1;
  }
}

// ----------------------------------------------------------------------------
// Kernel #9: Symplectic Generator Conditioning Vector Pack + Unit-Sphere Norm
// ----------------------------------------------------------------------------
struct GenCondParams {
  cond_dim: u32,
  bus_dim: u32,
  padded_in: u32,
  _pad: u32,
  t0: f32,
  t1: f32,
  t2: f32,
  t3: f32,
};

@group(0) @binding(0) var<uniform> gc_cfg: GenCondParams;
@group(0) @binding(1) var<storage, read> gc_h_cond: array<f32>;
@group(0) @binding(2) var<storage, read> gc_bus_ctx: array<f32>;
@group(0) @binding(3) var<storage, read_write> gc_raw_unit: array<f32>;

var<workgroup> sh_gc_sq: array<f32, 256>;

@compute @workgroup_size(256)
fn sf_pack_gen_cond(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let cd = gc_cfg.cond_dim;
  let bd = gc_cfg.bus_dim;
  let tot = gc_cfg.padded_in;
  var sq: f32 = 0.0;

  for (var i: u32 = tid; i < tot; i = i + 256u) {
    var val: f32 = 0.0;
    if (i < cd) {
      val = gc_h_cond[i];
    } else if (i < cd + bd) {
      val = gc_bus_ctx[i - cd];
    } else if (i == cd + bd) {
      val = gc_cfg.t0;
    } else if (i == cd + bd + 1u) {
      val = gc_cfg.t1;
    } else if (i == cd + bd + 2u) {
      val = gc_cfg.t2;
    } else if (i == cd + bd + 3u) {
      val = gc_cfg.t3;
    }
    gc_raw_unit[i] = val;
    sq = sq + val * val;
  }

  sh_gc_sq[tid] = sq;
  workgroupBarrier();
  for (var s: u32 = 128u; s > 0u; s = s >> 1u) {
    if (tid < s) {
      sh_gc_sq[tid] = sh_gc_sq[tid] + sh_gc_sq[tid + s];
    }
    workgroupBarrier();
  }
  let inv_n = 1.0 / max(1e-8, sqrt(sh_gc_sq[0]));
  for (var i: u32 = tid; i < tot; i = i + 256u) {
    gc_raw_unit[i] = gc_raw_unit[i] * inv_n;
  }
}

// ----------------------------------------------------------------------------
// Kernel #10: Horizon-Culled Symplectic Scar Resonance + `uv = [u_all, v_all]` Assembly
// ----------------------------------------------------------------------------
struct SymplecticUvParams {
  cond_dim: u32,
  target_dim: u32,
  base_rank: u32,
  num_scars: u32,
  active_slots: u32, // 4
  _p0: u32,
  _p1: u32,
  _p2: u32,
};

@group(0) @binding(0) var<uniform> suv_cfg: SymplecticUvParams;
@group(0) @binding(1) var<storage, read> suv_h_cond: array<f32>;        // [cond_dim]
@group(0) @binding(2) var<storage, read> suv_uv_e8: array<f32>;         // [2 * target_dim * base_rank]
@group(0) @binding(3) var<storage, read> suv_scar_poles: array<f32>;    // [num_scars, cond_dim] unit rows
@group(0) @binding(4) var<storage, read> suv_scar_kappa: array<f32>;    // [num_scars]
@group(0) @binding(5) var<storage, read> suv_scar_strength: array<f32>; // [num_scars]
@group(0) @binding(6) var<storage, read> suv_scar_u: array<f32>;        // [num_scars, target_dim] unit rows
@group(0) @binding(7) var<storage, read> suv_scar_v: array<f32>;        // [num_scars, target_dim] unit rows
@group(0) @binding(8) var<storage, read_write> suv_out: array<f32>;     // [target_dim, 2 * (base_rank + active_slots)]

var<workgroup> sh_eff_scars: array<f32, 256>;
var<workgroup> sh_top_idx: array<u32, 4>;
var<workgroup> sh_top_amp_u: array<f32, 4>;
var<workgroup> sh_top_amp_v: array<f32, 4>;
var<workgroup> sh_uv_inv1p: array<f32, 48>; // 2 * base_rank <= 48

@compute @workgroup_size(128)
fn sf_assemble_symplectic_uv(@builtin(local_invocation_id) lid: vec3<u32>) {
  let tid = lid.x;
  let cd = suv_cfg.cond_dim;
  let td = suv_cfg.target_dim;
  let rb = suv_cfg.base_rank;
  let ns = suv_cfg.num_scars;
  let act_m = suv_cfg.active_slots;
  let tot_r = rb + act_m;
  let two_tot_r = 2u * tot_r;

  // 1. Evaluate `C^1` spherical-cap resonance across all `num_scars <= 256`
  for (var e: u32 = tid; e < ns; e = e + 128u) {
    var cs: f32 = 0.0;
    let p_off = e * cd;
    for (var j: u32 = 0u; j < cd; j = j + 1u) {
      cs = cs + suv_h_cond[j] * suv_scar_poles[p_off + j];
    }
    let kap = clamp(suv_scar_kappa[e], -0.20, 0.92);
    let gate = c1_cap_scalar(cs, kap);
    sh_eff_scars[e] = gate * abs(suv_scar_strength[e]);
  }
  workgroupBarrier();

  // 2. Select Top-4 active horizon-resonant scars
  if (tid == 0u) {
    var used: array<u32, 4>;
    for (var m: u32 = 0u; m < act_m; m = m + 1u) {
      var best_g: f32 = -1.0;
      var best_e: u32 = 0u;
      for (var e: u32 = 0u; e < ns; e = e + 1u) {
        var skip = false;
        for (var prev: u32 = 0u; prev < m; prev = prev + 1u) {
          if (used[prev] == e) {
            skip = true;
          }
        }
        if (!skip && sh_eff_scars[e] > best_g) {
          best_g = sh_eff_scars[e];
          best_e = e;
        }
      }
      used[m] = best_e;
      sh_top_idx[m] = best_e;
      if (best_g > 0.0) {
        let amp = sqrt(best_g + 1e-12);
        let st_val = suv_scar_strength[best_e];
        let sgn = select(1.0, -1.0, st_val < 0.0);
        sh_top_amp_u[m] = amp * sgn;
        sh_top_amp_v[m] = amp;
      } else {
        sh_top_amp_u[m] = 0.0;
        sh_top_amp_v[m] = 0.0;
      }
    }
  }

  // 3. Compute `1 / sqrt(1 + ||uv_base[:, s, c]||_2^2)` along `target_dim` for each `(s, c)` in `[2, rb]`
  let two_rb = 2u * rb;
  if (tid < two_rb) {
    let s_half = tid / rb;
    let c_col = tid % rb;
    var n2: f32 = 0.0;
    let half_off = s_half * td * rb;
    for (var row: u32 = 0u; row < td; row = row + 1u) {
      let val = suv_uv_e8[half_off + row * rb + c_col];
      n2 = n2 + val * val;
    }
    sh_uv_inv1p[tid] = inverseSqrt(1.0 + n2);
  }
  workgroupBarrier();

  // 4. Write contiguous `[target_dim, 2 * tot_r]` (`[u_base, u_scar, v_base, v_scar]`)
  for (var row: u32 = tid; row < td; row = row + 128u) {
    let out_row_off = row * two_tot_r;
    let u_e8_off = row * rb;
    let v_e8_off = td * rb + row * rb;
    for (var c: u32 = 0u; c < rb; c = c + 1u) {
      suv_out[out_row_off + c] = suv_uv_e8[u_e8_off + c] * sh_uv_inv1p[c];
      suv_out[out_row_off + tot_r + c] = suv_uv_e8[v_e8_off + c] * sh_uv_inv1p[rb + c];
    }
    for (var m: u32 = 0u; m < act_m; m = m + 1u) {
      let sc_e = sh_top_idx[m];
      suv_out[out_row_off + rb + m] = suv_scar_u[sc_e * td + row] * sh_top_amp_u[m];
      suv_out[out_row_off + tot_r + rb + m] = suv_scar_v[sc_e * td + row] * sh_top_amp_v[m];
    }
  }
}
