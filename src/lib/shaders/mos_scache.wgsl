/**
 * WebGPU Shading Language (WGSL) Compute Kernels
 * Architecture: Mixture of Scars (MoS) & Scar-Cache (SCache)
 * 
 * Hardware-level tensor acceleration for in-browser client execution:
 * 1. SCache Hebbian Delta-Rule Matrix Update (O(1) 36.8 KB SRAM)
 * 2. SCache Associative Query with Directional Manifold Gating
 * 3. MoS Sparse Expert Low-Rank GEMM (x @ A @ B * gate * scale)
 * 4. MoS Top-K Competitive Router Logits
 */

// Constantes estruturais
const DIM: u32 = 96u;
const RANK: u32 = 8u;
const NUM_EXPERTS: u32 = 8u;

struct SCacheParams {
    dim: u32,
    num_slots: u32,
    decay: f32,
    learning_rate: f32,
    scale: f32,
    manifold_gate: f32,
};

// -----------------------------------------------------------------------------
// 1. SCACHE: HEBBIAN DELTA-RULE PARALLEL UPDATE
// W_{t+1} = gamma * W_t + eta * (v - W_t @ k) (x) k^T
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read_write> W_scar: array<f32>;        // [96 x 96]
@group(0) @binding(1) var<storage, read>       k_vec: array<f32>;         // [96]
@group(0) @binding(2) var<storage, read>       v_vec: array<f32>;         // [96]
@group(0) @binding(3) var<storage, read>       predicted_v: array<f32>;   // [96] (W_scar @ k)
@group(0) @binding(4) var<uniform>             params: SCacheParams;

@compute @workgroup_size(16, 16)
fn scache_delta_update(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    let col = global_id.y;

    if (row >= params.dim || col >= params.dim) {
        return;
    }

    let idx = row * params.dim + col;
    let error = v_vec[row] - predicted_v[row];
    let delta = params.learning_rate * error * k_vec[col];

    // Atualização com decaimento temporal gamma e produto externo
    W_scar[idx] = params.decay * W_scar[idx] + delta;
}

// -----------------------------------------------------------------------------
// 2. SCACHE: PARALLEL ASSOCIATIVE MATRIX-VECTOR QUERY
// r = (W_scar @ q) * max(0, cos(q, z_target))
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<storage, read>       W_scar_in: array<f32>;     // [96 x 96]
@group(0) @binding(1) var<storage, read>       q_vec: array<f32>;         // [96]
@group(0) @binding(2) var<storage, read_write> query_out: array<f32>;     // [96]
@group(0) @binding(3) var<uniform>             query_params: SCacheParams;

@compute @workgroup_size(64)
fn scache_associative_query(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let row = global_id.x;
    if (row >= query_params.dim) {
        return;
    }

    var sum: f32 = 0.0;
    let row_offset = row * query_params.dim;
    for (var col: u32 = 0u; col < query_params.dim; col = col + 1u) {
        sum = sum + W_scar_in[row_offset + col] * q_vec[col];
    }

    // Modulação pelo gating do manifold direcional
    query_out[row] = sum * query_params.scale * query_params.manifold_gate;
}

// -----------------------------------------------------------------------------
// 3. MOS: TOP-K COMPETITIVE ROUTER LOGITS
// logits[e] = (x @ R_e) / sqrt(dim)
// -----------------------------------------------------------------------------

struct MoSRouterParams {
    dim: u32,
    num_experts: u32,
    inv_sqrt_dim: f32,
};

@group(1) @binding(0) var<storage, read>       x_in: array<f32>;          // [96]
@group(1) @binding(1) var<storage, read>       router_weights: array<f32>;// [num_experts x 96]
@group(1) @binding(2) var<storage, read_write> router_logits: array<f32>; // [num_experts]
@group(1) @binding(3) var<uniform>             router_params: MoSRouterParams;

@compute @workgroup_size(8)
fn mos_router_logits(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let expert_id = global_id.x;
    if (expert_id >= router_params.num_experts) {
        return;
    }

    var dot: f32 = 0.0;
    let expert_offset = expert_id * router_params.dim;
    for (var i: u32 = 0u; i < router_params.dim; i = i + 1u) {
        dot = dot + x_in[i] * router_weights[expert_offset + i];
    }

    router_logits[expert_id] = dot * router_params.inv_sqrt_dim;
}

// -----------------------------------------------------------------------------
// 4. MOS: SPARSE EXPERT LOW-RANK PROJECTION
// delta = ((x @ A) @ B) * scale * max(0, cos(x, z_target))
// A: [dim x rank] (96 x 8)
// B: [rank x dim] (8 x 96)
// -----------------------------------------------------------------------------

struct MoSExpertParams {
    dim: u32,
    rank: u32,
    scale: f32,
    expert_gate: f32,
    routing_weight: f32,
};

@group(2) @binding(0) var<storage, read>       x_expert: array<f32>;      // [96]
@group(2) @binding(1) var<storage, read>       matrix_A: array<f32>;      // [96 x 8]
@group(2) @binding(2) var<storage, read>       matrix_B: array<f32>;      // [8 x 96]
@group(2) @binding(3) var<storage, read_write> expert_delta: array<f32>;  // [96]
@group(2) @binding(4) var<uniform>             expert_params: MoSExpertParams;

@compute @workgroup_size(64)
fn mos_expert_gemm(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let out_col = global_id.x;
    if (out_col >= expert_params.dim) {
        return;
    }

    // 1. Projeção intermediária: h[r] = sum_i (x[i] * A[i, r]) para r in [0, rank-1]
    // 2. Projeção final: delta[out_col] = sum_r (h[r] * B[r, out_col])
    var sum: f32 = 0.0;
    for (var r: u32 = 0u; r < expert_params.rank; r = r + 1u) {
        var h_r: f32 = 0.0;
        for (var i: u32 = 0u; i < expert_params.dim; i = i + 1u) {
            h_r = h_r + x_expert[i] * matrix_A[i * expert_params.rank + r];
        }
        sum = sum + h_r * matrix_B[r * expert_params.dim + out_col];
    }

    // Modulação por manifold gating direcional e peso do roteador Top-K
    let total_scale = expert_params.scale * expert_params.expert_gate * expert_params.routing_weight;
    expert_delta[out_col] = sum * total_scale;
}
