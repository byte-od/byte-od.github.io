/**
 * System 1 Engine — Standalone Web Evaluation Runtime
 * Portado para TypeScript para execução interativa no Portfólio.
 * Suporta Cubo Mágico 3x3 (Macro/CFOP) e CartPole-v1 com inferência via ONNX Runtime Web.
 * 
 * GitHub: https://github.com/byte-od/system_one
 */

// Interface para declaração do ONNX Runtime no escopo global
declare global {
  interface Window {
    ort?: any;
    System1?: any;
  }
}

/** Garante que o script do ONNX Runtime Web esteja carregado */
export async function ensureOnnxRuntime(): Promise<any> {
  if (typeof window !== 'undefined' && window.ort) {
    return window.ort;
  }

  return new Promise((resolve, reject) => {
    // Verifica se a tag já foi injetada
    const existing = document.querySelector('script[src*="onnxruntime-web"]');
    if (existing) {
      existing.addEventListener('load', () => resolve(window.ort));
      existing.addEventListener('error', (e) => reject(e));
      return;
    }

    const script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.min.js';
    script.async = true;
    script.onload = () => resolve(window.ort);
    script.onerror = (err) => reject(err);
    document.head.appendChild(script);
  });
}

// =====================================================================
// SIMULADOR DO CUBO MÁGICO 3X3 (MACRO / CFOP)
// =====================================================================

export class RubiksCubeSim {
  static ATOMIC_MOVES = [
    'U', 'U_prime', 'D', 'D_prime',
    'F', 'F_prime', 'B', 'B_prime',
    'R', 'R_prime', 'L', 'L_prime',
  ];

  static MACRO_ACTIONS: Record<string, string[]> = {
    SEXY_MOVE_R: ['R', 'U', 'R_prime', 'U_prime'],
    SEXY_MOVE_R_PRIME: ['U', 'R', 'U_prime', 'R_prime'],
    SEXY_MOVE_L: ['L_prime', 'U_prime', 'L', 'U'],
    SEXY_MOVE_L_PRIME: ['U_prime', 'L_prime', 'U', 'L'],
    SUNE: ['R', 'U', 'R_prime', 'U', 'R', 'U', 'U', 'R_prime'],
    ANTI_SUNE: ['R', 'U', 'U', 'R_prime', 'U_prime', 'R', 'U_prime', 'R_prime'],
    YELLOW_CROSS: ['F', 'R', 'U', 'R_prime', 'U_prime', 'F_prime'],
    YELLOW_CROSS_PRIME: ['F', 'U', 'R', 'U_prime', 'R_prime', 'F_prime'],
    ROTATE_Y: ['Y'],
    ROTATE_Y_PRIME: ['Y_prime'],
    U_TURN: ['U'],
    U_PRIME_TURN: ['U_prime'],
  };

  static INVERSE_ATOMIC: Record<string, string> = {
    U: 'U_prime', U_prime: 'U',
    D: 'D_prime', D_prime: 'D',
    F: 'F_prime', F_prime: 'F',
    B: 'B_prime', B_prime: 'B',
    R: 'R_prime', R_prime: 'R',
    L: 'L_prime', L_prime: 'L',
  };

  static INVERSE_MACROS: Record<string, string> = {
    SEXY_MOVE_R: 'SEXY_MOVE_R_PRIME',
    SEXY_MOVE_R_PRIME: 'SEXY_MOVE_R',
    SEXY_MOVE_L: 'SEXY_MOVE_L_PRIME',
    SEXY_MOVE_L_PRIME: 'SEXY_MOVE_L',
    SUNE: 'ANTI_SUNE',
    ANTI_SUNE: 'SUNE',
    YELLOW_CROSS: 'YELLOW_CROSS_PRIME',
    YELLOW_CROSS_PRIME: 'YELLOW_CROSS',
    ROTATE_Y: 'ROTATE_Y_PRIME',
    ROTATE_Y_PRIME: 'ROTATE_Y',
    U_TURN: 'U_PRIME_TURN',
    U_PRIME_TURN: 'U_TURN',
  };

  static SCRAMBLE_MACRO_NAMES = [
    'SEXY_MOVE_R', 'SEXY_MOVE_R_PRIME',
    'SEXY_MOVE_L', 'SEXY_MOVE_L_PRIME',
    'SUNE', 'ANTI_SUNE',
    'YELLOW_CROSS', 'YELLOW_CROSS_PRIME',
    'U_TURN', 'U_PRIME_TURN',
  ];

  static MACRO_NAMES = Object.keys(RubiksCubeSim.MACRO_ACTIONS);

  static PALETTE: Record<number, string> = {
    0: '#F5F5FA', // U: Branco
    1: '#FACC15', // D: Amarelo
    2: '#22C55E', // F: Verde
    3: '#3B82F6', // B: Azul
    4: '#EF4444', // R: Vermelho
    5: '#F97316', // L: Laranja
  };

  static FACE_LAYOUT: Record<number, { x: number; y: number }> = {
    0: { x: 164, y: 46 },  // U
    5: { x: 88,  y: 122 }, // L
    2: { x: 164, y: 122 }, // F
    4: { x: 240, y: 122 }, // R
    3: { x: 316, y: 122 }, // B
    1: { x: 164, y: 198 }, // D
  };

  permutations: Record<string, Int32Array>;
  state: Int32Array = new Int32Array(54);
  lastAction: string = 'RESET';
  steps: number = 0;

  constructor() {
    this.permutations = this._initPermutations();
    this.reset();
  }

  private _initPermutations(): Record<string, Int32Array> {
    const rotF = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]];
    const rotB = [[0, -1, 0], [1, 0, 0], [0, 0, 1]];
    const rotR = [[1, 0, 0], [0, 0, -1], [0, 1, 0]];
    const rotL = [[1, 0, 0], [0, 0, 1], [0, -1, 0]];
    const rotU = [[0, 0, -1], [0, 1, 0], [1, 0, 0]];
    const rotD = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]];

    const transpose = (m: number[][]) => [
      [m[0][0], m[1][0], m[2][0]],
      [m[0][1], m[1][1], m[2][1]],
      [m[0][2], m[1][2], m[2][2]],
    ];

    const moves: Record<string, { m: number[][]; cond: (p: number[]) => boolean }> = {
      F: { m: rotF, cond: (p) => p[2] === 1 },
      F_prime: { m: transpose(rotF), cond: (p) => p[2] === 1 },
      B: { m: rotB, cond: (p) => p[2] === -1 },
      B_prime: { m: transpose(rotB), cond: (p) => p[2] === -1 },
      R: { m: rotR, cond: (p) => p[0] === 1 },
      R_prime: { m: transpose(rotR), cond: (p) => p[0] === 1 },
      L: { m: rotL, cond: (p) => p[0] === -1 },
      L_prime: { m: transpose(rotL), cond: (p) => p[0] === -1 },
      U: { m: rotU, cond: (p) => p[1] === 1 },
      U_prime: { m: transpose(rotU), cond: (p) => p[1] === 1 },
      D: { m: rotD, cond: (p) => p[1] === -1 },
      D_prime: { m: transpose(rotD), cond: (p) => p[1] === -1 },
      Y: { m: rotU, cond: () => true },
      Y_prime: { m: transpose(rotU), cond: () => true },
    };

    const getSlot = (pos: number[], norm: number[]) => {
      const [x, y, z] = pos;
      const [nx, ny, nz] = norm;
      let face = 0, r = 0, c = 0;
      if (ny === 1) { face = 0; r = 1 - z; c = x + 1; }
      else if (ny === -1) { face = 1; r = z + 1; c = x + 1; }
      else if (nz === 1) { face = 2; r = 1 - y; c = x + 1; }
      else if (nz === -1) { face = 3; r = 1 - y; c = 1 - x; }
      else if (nx === 1) { face = 4; r = 1 - y; c = 1 - z; }
      else if (nx === -1) { face = 5; r = 1 - y; c = z + 1; }
      return face * 9 + (r * 3 + c);
    };

    const perms: Record<string, Int32Array> = {};
    for (const [mName, { m: rotMat, cond }] of Object.entries(moves)) {
      const stickers: Array<{ pos: number[]; norm: number[]; srcSlot?: number }> = [];
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1, 1 - r], norm: [0, 1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, -1, r - 1], norm: [0, -1, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [c - 1, 1 - r, 1], norm: [0, 0, 1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1 - c, 1 - r, -1], norm: [0, 0, -1] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [1, 1 - r, 1 - c], norm: [1, 0, 0] });
      for (let r = 0; r < 3; r++) for (let c = 0; c < 3; c++) stickers.push({ pos: [-1, 1 - r, c - 1], norm: [-1, 0, 0] });

      for (let i = 0; i < stickers.length; i++) {
        stickers[i].srcSlot = getSlot(stickers[i].pos, stickers[i].norm);
      }

      const mul = (mat: number[][], v: number[]) => [
        mat[0][0] * v[0] + mat[0][1] * v[1] + mat[0][2] * v[2],
        mat[1][0] * v[0] + mat[1][1] * v[1] + mat[1][2] * v[2],
        mat[2][0] * v[0] + mat[2][1] * v[1] + mat[2][2] * v[2],
      ];

      const perm = new Int32Array(54);
      for (const s of stickers) {
        let p = s.pos, n = s.norm;
        if (cond(p)) { p = mul(rotMat, p); n = mul(rotMat, n); }
        const destSlot = getSlot(p, n);
        perm[destSlot] = s.srcSlot!;
      }
      perms[mName] = perm;
    }
    return perms;
  }

  static AXES_MOVES: Record<string, string[]> = {
    X: ['R', 'R_prime', 'L', 'L_prime'],
    Y: ['U', 'U_prime', 'D', 'D_prime'],
    Z: ['F', 'F_prime', 'B', 'B_prime'],
  };

  scrambleSequence: string[] = [];
  scrambleBaseline = 54;

  reset() {
    this.state = new Int32Array(54);
    for (let f = 0; f < 6; f++) {
      for (let i = 0; i < 9; i++) this.state[f * 9 + i] = f;
    }
    this.lastAction = 'RESET';
    this.scrambleSequence = [];
    this.steps = 0;
    this.scrambleBaseline = 54;
  }

  applyAtomic(moveName: string, countStep: boolean = true) {
    const perm = this.permutations[moveName];
    if (!perm) return;
    const nextState = new Int32Array(54);
    for (let i = 0; i < 54; i++) nextState[i] = this.state[perm[i]];
    this.state = nextState;
    if (countStep) {
      this.lastAction = moveName;
      this.steps++;
    }
  }

  predictStateHash(moveName: string): string {
    const perm = this.permutations[moveName];
    if (!perm) return '';
    const nextState = new Int32Array(54);
    for (let i = 0; i < 54; i++) nextState[i] = this.state[perm[i]];
    return nextState.join('');
  }

  getStateHash(): string {
    return this.state.join('');
  }

  applyMacro(macroName: string) {
    const seq = RubiksCubeSim.MACRO_ACTIONS[macroName];
    if (!seq) return;
    for (const m of seq) this.applyAtomic(m, false);
    this.lastAction = macroName;
    this.steps++;
  }

  scrambleAtomic(depth = 4): string[] {
    const moves: string[] = [];
    let lastAxis: string | null = null;
    const targetDepth = Math.max(1, depth);
    const axes: string[] = ['X', 'Y', 'Z'];

    while (moves.length < targetDepth) {
      const availableAxes: string[] = lastAxis ? axes.filter((a) => a !== lastAxis) : axes;
      const axis: string = availableAxes[Math.floor(Math.random() * availableAxes.length)];
      const choices: string[] = RubiksCubeSim.AXES_MOVES[axis] || RubiksCubeSim.ATOMIC_MOVES;
      const move = choices[Math.floor(Math.random() * choices.length)];

      this.applyAtomic(move, false);
      moves.push(move);
      lastAxis = axis;
    }
    return moves;
  }

  scrambleMacro(depth = 2): string[] {
    const moves: string[] = [];
    let lastMove: string | null = null;
    const candidates = RubiksCubeSim.SCRAMBLE_MACRO_NAMES;
    const targetDepth = Math.max(1, depth);
    let attempts = 0;
    const maxAttempts = Math.max(100, targetDepth * 10);

    while ((moves.length < targetDepth || this.isSolved()) && attempts < maxAttempts) {
      attempts++;
      const filtered = candidates.filter(
        (m) => m !== RubiksCubeSim.INVERSE_MACROS[lastMove || ''] && m !== lastMove
      );
      const act = filtered[Math.floor(Math.random() * filtered.length)] || candidates[0];
      this.applyMacro(act);
      moves.push(act);
      lastMove = act;
    }
    return moves;
  }

  scramble(depth = 4, mode = 'atomic'): string[] {
    for (let attempt = 0; attempt < 20; attempt++) {
      this.reset();
      const moves = mode === 'macro' ? this.scrambleMacro(depth) : this.scrambleAtomic(depth);
      if (this.getRawAlignedCount() < 54 && !this.isSolved()) {
        this.lastAction = 'EMBARALHADO';
        this.scrambleSequence = moves;
        this.steps = 0;
        this.scrambleBaseline = this.getRawAlignedCount();
        return moves;
      }
    }
    this.reset();
    const fallback = ['R', 'U', 'F_prime', 'L'];
    for (const m of fallback) this.applyAtomic(m);
    this.lastAction = 'EMBARALHADO';
    this.scrambleSequence = fallback;
    this.steps = 0;
    this.scrambleBaseline = this.getRawAlignedCount();
    return fallback;
  }

  getRawAlignedCount(): number {
    let count = 0;
    for (let f = 0; f < 6; f++) {
      const center = this.state[f * 9 + 4];
      for (let i = 0; i < 9; i++) {
        if (this.state[f * 9 + i] === center) count++;
      }
    }
    return count;
  }

  getAlignedCount(): number {
    if (this.isSolved()) return 54;
    const raw = this.getRawAlignedCount();
    const base = this.scrambleBaseline;
    if (raw <= base) return 0;
    const progress = (raw - base) / (54 - base);
    return Math.min(54, Math.max(0, Math.round(progress * 54)));
  }

  getScore(): number {
    const raw = this.getRawAlignedCount();
    return Math.max(0.0, Math.min(1.0, (raw - 6) / 48.0));
  }

  getScorePct(): number {
    if (this.isSolved()) return 100.0;
    const raw = this.getRawAlignedCount();
    const base = this.scrambleBaseline;
    if (raw <= base) return 0.0;
    const progress = (raw - base) / (54 - base);
    return Math.min(100.0, Math.max(0.0, progress * 100));
  }

  isSolved(): boolean {
    return this.getRawAlignedCount() === 54;
  }

  getOneHot(): Float32Array {
    const arr = new Float32Array(324);
    for (let i = 0; i < 54; i++) {
      const color = this.state[i];
      arr[i * 6 + color] = 1.0;
    }
    return arr;
  }

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;

    // Fundo Cyber Dark
    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, width, height);

    // Moldura
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    // Cabeçalho
    const aligned = this.getAlignedCount();
    const scorePct = this.getScorePct().toFixed(1);
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | CUBO MÁGICO 3X3', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    let statusLine = `Passos: ${this.steps} | Alinhamento: ${aligned}/54 (${scorePct}%)`;
    if (this.scrambleSequence && this.scrambleSequence.length > 0 && this.steps === 0) {
      statusLine += ` | Scramble: ${this.scrambleSequence.join(' ')}`;
    }
    ctx.fillText(statusLine, 20, 42);

    // Desenho das 6 faces em Cruz (desdobramento sincronizado com o modelo 3D)
    for (const [fIdxStr, { x: fx, y: fy }] of Object.entries(RubiksCubeSim.FACE_LAYOUT)) {
      const fIdx = parseInt(fIdxStr, 10);
      for (let r = 0; r < 3; r++) {
        for (let c = 0; c < 3; c++) {
          const stickerIdx = fIdx * 9 + (r * 3 + c);
          const colId = this.state[stickerIdx];
          ctx.fillStyle = RubiksCubeSim.PALETTE[colId] || '#ffffff';
          // U (Face 0) e D (Face 1): invertem linha para que a borda com F fique adjacente no plano
          const drawRow = (fIdx === 0 || fIdx === 1) ? (2 - r) : r;
          const x1 = fx + c * 24;
          const y1 = fy + drawRow * 24;
          ctx.fillRect(x1, y1, 22, 22);
          ctx.strokeStyle = '#0f172a';
          ctx.lineWidth = 1;
          ctx.strokeRect(x1, y1, 22, 22);
        }
      }
    }

    // Rodapé
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    ctx.fillStyle = this.isSolved() ? '#4ade80' : '#e2e8f0';
    ctx.fillText(
      this.isSolved() ? 'STATUS: RESOLVIDO 🏆' : `AÇÃO ATIVA: ${this.lastAction}`,
      20,
      height - 16
    );
  }
}

// =====================================================================
// SIMULADOR CARTPOLE-V1
// =====================================================================

export class CartPoleSim {
  gravity = 9.8;
  masscart = 1.0;
  masspole = 0.1;
  total_mass = 1.1;
  length = 0.5;
  polemass_length = 0.05;
  force_mag = 10.0;
  tau = 0.02;

  x = 0;
  x_dot = 0;
  theta = 0;
  theta_dot = 0;
  steps = 0;

  constructor() {
    this.reset();
  }

  reset() {
    this.x = (Math.random() - 0.5) * 0.1;
    this.x_dot = (Math.random() - 0.5) * 0.1;
    this.theta = (Math.random() - 0.5) * 0.1;
    this.theta_dot = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
  }

  getState(): Float32Array {
    return new Float32Array([this.x, this.x_dot, this.theta, this.theta_dot]);
  }

  step(action: number) {
    const force = action === 1 ? this.force_mag : -this.force_mag;
    const costh = Math.cos(this.theta);
    const sinth = Math.sin(this.theta);

    const temp = (force + this.polemass_length * this.theta_dot * this.theta_dot * sinth) / this.total_mass;
    const thetaacc =
      (this.gravity * sinth - costh * temp) /
      (this.length * (4.0 / 3.0 - (this.masspole * costh * costh) / this.total_mass));
    const xacc = temp - (this.polemass_length * thetaacc * costh) / this.total_mass;

    this.x += this.tau * this.x_dot;
    this.x_dot += this.tau * xacc;
    this.theta += this.tau * this.theta_dot;
    this.theta_dot += this.tau * thetaacc;
    this.steps++;

    const done = Math.abs(this.x) > 2.4 || Math.abs(this.theta) > 0.2095;
    return { done, steps: this.steps };
  }

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const width = canvas.width;
    const height = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, width, height);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, width - 12, height - 12);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | CARTPOLE-V1', 20, 26);
    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`Passos Equilibrados: ${this.steps}`, 20, 42);

    // Linha do solo
    const groundY = height - 80;
    ctx.strokeStyle = '#334155';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(20, groundY);
    ctx.lineTo(width - 20, groundY);
    ctx.stroke();

    // Carrinho
    const scale = width / 4.8;
    const cartX = width / 2 + this.x * scale;
    const cartW = 60, cartH = 30;
    ctx.fillStyle = '#38bdf8';
    ctx.fillRect(cartX - cartW / 2, groundY - cartH, cartW, cartH);

    // Haste
    const poleLen = 90;
    const poleEndX = cartX + Math.sin(this.theta) * poleLen;
    const poleEndY = (groundY - cartH) - Math.cos(this.theta) * poleLen;
    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(cartX, groundY - cartH);
    ctx.lineTo(poleEndX, poleEndY);
    ctx.stroke();
  }
}

// =====================================================================
// SIMULADOR LUNARLANDER-V3 (POUSO ESPACIAL)
// =====================================================================

export class LunarLanderSim {
  groundY = 0.18;
  x = 0;
  y = 1.3;
  vx = 0;
  vy = 0;
  angle = 0;
  v_angle = 0;
  leftContact = 0;
  rightContact = 0;
  steps = 0;
  fuel = 100;
  lastAction = 0;
  isLanded = false;
  isCrashed = false;
  particles: Array<{ x: number; y: number; vx: number; vy: number; life: number; color: string }> = [];

  constructor() {
    this.reset();
  }

  reset() {
    this.x = (Math.random() - 0.5) * 0.4;
    this.y = 1.3 + Math.random() * 0.1;
    this.vx = (Math.random() - 0.5) * 0.2;
    this.vy = -(0.02 + Math.random() * 0.04);
    this.angle = (Math.random() - 0.5) * 0.1;
    this.v_angle = (Math.random() - 0.5) * 0.05;
    this.leftContact = 0;
    this.rightContact = 0;
    this.steps = 0;
    this.fuel = 100;
    this.lastAction = 0;
    this.isLanded = false;
    this.isCrashed = false;
    this.particles = [];
  }

  getState(): Float32Array {
    return new Float32Array([
      this.x,
      this.y,
      this.vx,
      this.vy,
      this.angle,
      this.v_angle,
      this.leftContact,
      this.rightContact,
    ]);
  }

  step(action: number): { done: boolean; steps: number } {
    this.lastAction = action;
    this.steps++;

    const dt = 0.02;
    const gravity = -3.2; // Gravidade lunar adaptada
    const mainThrust = 9.0;
    const sideThrust = 2.4;

    let thrustX = 0;
    let thrustY = gravity;
    let torque = 0;

    // Ações: 0=Noop, 1=Propulsor Esquerdo, 2=Principal, 3=Propulsor Direito
    if (action === 2 && this.fuel > 0) {
      thrustX -= Math.sin(this.angle) * mainThrust;
      thrustY += Math.cos(this.angle) * mainThrust;
      this.fuel = Math.max(0, this.fuel - 0.12);

      for (let i = 0; i < 3; i++) {
        this.particles.push({
          x: this.x - Math.sin(this.angle) * 0.08,
          y: this.y - Math.cos(this.angle) * 0.08,
          vx: -Math.sin(this.angle) * (1.2 + Math.random()) + (Math.random() - 0.5) * 0.3,
          vy: -Math.cos(this.angle) * (1.2 + Math.random()) + (Math.random() - 0.5) * 0.3,
          life: 1.0,
          color: Math.random() > 0.4 ? '#f59e0b' : '#ef4444',
        });
      }
    } else if (action === 1 && this.fuel > 0) {
      torque -= sideThrust;
      thrustX += Math.cos(this.angle) * 0.6;
      this.fuel = Math.max(0, this.fuel - 0.06);

      for (let i = 0; i < 2; i++) {
        this.particles.push({
          x: this.x - Math.cos(this.angle) * 0.06,
          y: this.y + Math.sin(this.angle) * 0.06,
          vx: -Math.cos(this.angle) * 0.7,
          vy: Math.sin(this.angle) * 0.7,
          life: 0.8,
          color: '#38bdf8',
        });
      }
    } else if (action === 3 && this.fuel > 0) {
      torque += sideThrust;
      thrustX -= Math.cos(this.angle) * 0.6;
      this.fuel = Math.max(0, this.fuel - 0.06);

      for (let i = 0; i < 2; i++) {
        this.particles.push({
          x: this.x + Math.cos(this.angle) * 0.06,
          y: this.y - Math.sin(this.angle) * 0.06,
          vx: Math.cos(this.angle) * 0.7,
          vy: -Math.sin(this.angle) * 0.7,
          life: 0.8,
          color: '#38bdf8',
        });
      }
    }

    this.vx += thrustX * dt;
    this.vy += thrustY * dt;
    this.v_angle += torque * dt;

    this.vx *= 0.998;
    this.v_angle *= 0.96;

    this.x += this.vx * dt;
    this.y += this.vy * dt;
    this.angle += this.v_angle * dt;

    // Atualiza partículas
    for (let i = this.particles.length - 1; i >= 0; i--) {
      const p = this.particles[i];
      p.x += p.vx * dt;
      p.y += p.vy * dt;
      p.life -= dt * 2.5;
      if (p.life <= 0) this.particles.splice(i, 1);
    }

    let done = false;

    if (this.y <= this.groundY) {
      this.y = this.groundY;
      const speed = Math.hypot(this.vx, this.vy);
      const angleDeg = Math.abs((this.angle * 180) / Math.PI);
      const onPad = Math.abs(this.x) < 0.28;

      if (speed < 0.45 && angleDeg < 15 && onPad) {
        this.isLanded = true;
        this.leftContact = 1;
        this.rightContact = 1;
        this.vx = 0;
        this.vy = 0;
        this.v_angle = 0;
        done = true;
      } else {
        this.isCrashed = true;
        done = true;
      }
    }

    if (Math.abs(this.x) > 1.3 || this.y > 1.9) {
      done = true;
    }

    return { done, steps: this.steps };
  }

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    // Céu Espacial Noturno
    ctx.fillStyle = '#06080e';
    ctx.fillRect(0, 0, w, h);

    // Estrelas de fundo
    ctx.fillStyle = '#64748b';
    for (let i = 0; i < 30; i++) {
      const sx = (Math.sin(i * 12.3) * 0.5 + 0.5) * w;
      const sy = (Math.cos(i * 7.7) * 0.5 + 0.5) * (h * 0.65);
      ctx.fillRect(sx, sy, 1.5, 1.5);
    }

    // Superfície Lunar & Plataforma de Pouso
    const groundScreenY = h - 55;
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.moveTo(0, h);
    ctx.lineTo(0, groundScreenY + 15);
    ctx.lineTo(w * 0.35, groundScreenY + 5);
    ctx.lineTo(w * 0.4, groundScreenY);
    ctx.lineTo(w * 0.6, groundScreenY);
    ctx.lineTo(w * 0.65, groundScreenY + 8);
    ctx.lineTo(w, groundScreenY + 12);
    ctx.lineTo(w, h);
    ctx.fill();

    // Plataforma de pouso verde neon
    ctx.strokeStyle = '#22c55e';
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(w * 0.38, groundScreenY);
    ctx.lineTo(w * 0.62, groundScreenY);
    ctx.stroke();

    // Bandeiras da plataforma
    ctx.fillStyle = '#22c55e';
    ctx.fillRect(w * 0.38, groundScreenY - 14, 2, 14);
    ctx.fillRect(w * 0.38 + 2, groundScreenY - 14, 8, 6);
    ctx.fillRect(w * 0.62, groundScreenY - 14, 2, 14);
    ctx.fillRect(w * 0.62 + 2, groundScreenY - 14, 8, 6);

    // Mapeamento de coordenadas
    const toScreenX = (valX: number) => w / 2 + valX * (w * 0.42);
    const toScreenY = (valY: number) => groundScreenY - valY * (h * 0.45);

    // Renderiza partículas de propulsão
    for (const p of this.particles) {
      ctx.fillStyle = p.color;
      ctx.globalAlpha = Math.max(0, p.life);
      ctx.beginPath();
      ctx.arc(toScreenX(p.x), toScreenY(p.y), 3 * p.life, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1.0;

    // Renderiza Módulo Lunar
    const lx = toScreenX(this.x);
    const ly = toScreenY(this.y);

    ctx.save();
    ctx.translate(lx, ly);
    ctx.rotate(this.angle);

    // Pernas de pouso
    ctx.strokeStyle = '#94a3b8';
    ctx.lineWidth = 2;
    ctx.beginPath();
    // Esquerda
    ctx.moveTo(-10, 6);
    ctx.lineTo(-18, 18);
    ctx.lineTo(-24, 18);
    // Direita
    ctx.moveTo(10, 6);
    ctx.lineTo(18, 18);
    ctx.lineTo(24, 18);
    ctx.stroke();

    // Corpo do Módulo (Hexágono espacial)
    ctx.fillStyle = this.isLanded ? '#22c55e' : this.isCrashed ? '#ef4444' : '#38bdf8';
    ctx.beginPath();
    ctx.moveTo(-12, 6);
    ctx.lineTo(12, 6);
    ctx.lineTo(16, -6);
    ctx.lineTo(8, -16);
    ctx.lineTo(-8, -16);
    ctx.lineTo(-16, -6);
    ctx.closePath();
    ctx.fill();
    ctx.strokeStyle = '#0284c7';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Janela do cockpit
    ctx.fillStyle = '#0f172a';
    ctx.beginPath();
    ctx.arc(0, -6, 5, 0, Math.PI * 2);
    ctx.fill();
    ctx.strokeStyle = '#38bdf8';
    ctx.stroke();

    ctx.restore();

    // Moldura Cyber & Cabeçalho
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | LUNARLANDER-V3', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    const alt = Math.max(0, (this.y - this.groundY) * 100).toFixed(0);
    const vel = (Math.hypot(this.vx, this.vy) * 10).toFixed(1);
    ctx.fillText(`Passos: ${this.steps} | Altitude: ${alt}m | Vel: ${vel}m/s | Combustível: ${this.fuel.toFixed(0)}%`, 20, 42);

    // Rodapé
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    if (this.isLanded) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: POUSO PERFEITO 🏆', 20, h - 16);
    } else if (this.isCrashed) {
      ctx.fillStyle = '#ef4444';
      ctx.fillText('STATUS: IMPACTO CRÍTICO 💥 (REINICIANDO)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const actionNames = ['ORBITANDO (NOOP)', 'PROPULSOR ESQUERDO ◀', 'PROPULSOR PRINCIPAL ▲', 'PROPULSOR DIREITO ▶'];
      ctx.fillText(`AÇÃO: ${actionNames[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// SIMULADOR MOUNTAINCAR-V0 (CONTROLE NÃO-LINEAR DE INÉRCIA)
// =====================================================================

export class MountainCarSim {
  minPosition = -1.2;
  maxPosition = 0.6;
  maxSpeed = 0.07;
  goalPosition = 0.5;
  power = 0.0015;
  gravity = 0.0025;

  position = -0.5;
  velocity = 0;
  steps = 0;
  lastAction = 1;
  isGoalReached = false;

  constructor() {
    this.reset();
  }

  reset() {
    this.position = -0.6 + Math.random() * 0.2;
    this.velocity = 0;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
  }

  getState(): Float32Array {
    return new Float32Array([this.position, this.velocity]);
  }

  step(action: number): { done: boolean; steps: number } {
    this.lastAction = action;
    this.steps++;

    // Ações: 0=Acelera Esquerda (-1), 1=Neutro (0), 2=Acelera Direita (+1)
    const force = action === 0 ? -1 : action === 2 ? 1 : 0;
    this.velocity += force * this.power + Math.cos(3 * this.position) * -this.gravity;
    this.velocity = Math.max(-this.maxSpeed, Math.min(this.maxSpeed, this.velocity));

    this.position += this.velocity;
    this.position = Math.max(this.minPosition, Math.min(this.maxPosition, this.position));

    if (this.position === this.minPosition && this.velocity < 0) {
      this.velocity = 0;
    }

    const done = this.position >= this.goalPosition;
    if (done) this.isGoalReached = true;

    return { done, steps: this.steps };
  }

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    // Mapeamento de posições
    const toScreenX = (pos: number) => {
      return ((pos - this.minPosition) / (this.maxPosition - this.minPosition)) * (w - 60) + 30;
    };

    const toScreenY = (pos: number) => {
      const heightNorm = Math.sin(3 * pos); // [-1, 1]
      return h - 70 - heightNorm * 80;
    };

    // Desenha o vale montanhoso sinusoidal
    ctx.beginPath();
    ctx.moveTo(30, h - 30);
    for (let px = this.minPosition; px <= this.maxPosition; px += 0.02) {
      ctx.lineTo(toScreenX(px), toScreenY(px));
    }
    ctx.lineTo(w - 30, h - 30);
    ctx.closePath();
    ctx.fillStyle = '#0f172a';
    ctx.fill();

    // Linha de contorno neon do relevo
    ctx.beginPath();
    for (let px = this.minPosition; px <= this.maxPosition; px += 0.02) {
      const sx = toScreenX(px);
      const sy = toScreenY(px);
      if (px === this.minPosition) ctx.moveTo(sx, sy);
      else ctx.lineTo(sx, sy);
    }
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Bandeira de objetivo (Meta x = 0.5)
    const goalX = toScreenX(this.goalPosition);
    const goalY = toScreenY(this.goalPosition);

    ctx.strokeStyle = '#f59e0b';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(goalX, goalY);
    ctx.lineTo(goalX, goalY - 32);
    ctx.stroke();

    ctx.fillStyle = '#eab308';
    ctx.beginPath();
    ctx.moveTo(goalX, goalY - 32);
    ctx.lineTo(goalX + 16, goalY - 24);
    ctx.lineTo(goalX, goalY - 16);
    ctx.closePath();
    ctx.fill();

    // Renderiza o Carro / Rover
    const carX = toScreenX(this.position);
    const carY = toScreenY(this.position);
    const slope = Math.cos(3 * this.position);
    const carAngle = Math.atan(slope);

    ctx.save();
    ctx.translate(carX, carY);
    ctx.rotate(carAngle);

    // Chassi do carro
    ctx.fillStyle = this.isGoalReached ? '#22c55e' : '#38bdf8';
    ctx.fillRect(-15, -12, 30, 10);

    // Rodas
    ctx.fillStyle = '#0284c7';
    ctx.beginPath();
    ctx.arc(-10, -2, 4, 0, Math.PI * 2);
    ctx.arc(10, -2, 4, 0, Math.PI * 2);
    ctx.fill();

    // Luz / Farol do Rover
    ctx.fillStyle = '#facc15';
    ctx.fillRect(13, -10, 3, 5);

    ctx.restore();

    // Cabeçalho
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | MOUNTAINCAR-V0', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    const energyKin = (0.5 * this.velocity * this.velocity * 1000).toFixed(1);
    ctx.fillText(`Passos: ${this.steps} | Posição: ${this.position.toFixed(2)} | Inércia: ${energyKin}J`, 20, 42);

    // Rodapé
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    if (this.isGoalReached) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: OBJETIVO ALCANÇADO 🏆 (TOPO DO VALE)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const actNames = ['IMPULSO ESQUERDA ◀', 'EMBALO LIVRE (NEUTRO)', 'IMPULSO DIREITA ▶'];
      ctx.fillText(`AÇÃO: ${actNames[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// SIMULADOR ACROBOT-V1 (PÊNDULO DUPLO ACROBÁTICO)
// =====================================================================

export class AcrobotSim {
  th1 = 0;
  th2 = 0;
  dth1 = 0;
  dth2 = 0;
  steps = 0;
  lastAction = 1;
  isGoalReached = false;
  trail: Array<{ x: number; y: number }> = [];

  constructor() {
    this.reset();
  }

  reset() {
    this.th1 = (Math.random() - 0.5) * 0.2;
    this.th2 = (Math.random() - 0.5) * 0.2;
    this.dth1 = (Math.random() - 0.5) * 0.1;
    this.dth2 = (Math.random() - 0.5) * 0.1;
    this.steps = 0;
    this.lastAction = 1;
    this.isGoalReached = false;
    this.trail = [];
  }

  getState(): Float32Array {
    return new Float32Array([
      Math.cos(this.th1),
      Math.sin(this.th1),
      Math.cos(this.th2),
      Math.sin(this.th2),
      this.dth1,
      this.dth2,
    ]);
  }

  step(action: number): { done: boolean; steps: number } {
    this.lastAction = action;
    this.steps++;

    const dt = 0.05;
    const torque = action === 0 ? -1.0 : action === 2 ? 1.0 : 0.0;
    const g = 9.8;

    // Física simplificada e estável do duplo pêndulo
    const dth1_acc = -g * Math.sin(this.th1) * 0.5 + torque * 0.4;
    const dth2_acc = -g * Math.sin(this.th1 + this.th2) * 0.5 - torque * 0.8;

    this.dth1 += dth1_acc * dt;
    this.dth2 += dth2_acc * dt;

    this.dth1 *= 0.995;
    this.dth2 *= 0.995;

    this.th1 += this.dth1 * dt;
    this.th2 += this.dth2 * dt;

    // Meta: ponta atinge acima da linha de altura 1.0 (ou cos(th1) + cos(th1 + th2) < -1.0)
    const tipHeight = -(Math.cos(this.th1) + Math.cos(this.th1 + this.th2));
    const done = tipHeight > 1.0;
    if (done) this.isGoalReached = true;

    return { done, steps: this.steps };
  }

  render(canvas: HTMLCanvasElement) {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const w = canvas.width;
    const h = canvas.height;

    ctx.fillStyle = '#07090e';
    ctx.fillRect(0, 0, w, h);

    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.strokeRect(6, 6, w - 12, h - 12);

    const pivotX = w / 2;
    const pivotY = h / 2 + 30;
    const l1 = 55;
    const l2 = 55;

    const jointX = pivotX + l1 * Math.sin(this.th1);
    const jointY = pivotY + l1 * Math.cos(this.th1);

    const tipX = jointX + l2 * Math.sin(this.th1 + this.th2);
    const tipY = jointY + l2 * Math.cos(this.th1 + this.th2);

    // Rastro da ponta
    this.trail.push({ x: tipX, y: tipY });
    if (this.trail.length > 25) this.trail.shift();

    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < this.trail.length; i++) {
      const p = this.trail[i];
      ctx.globalAlpha = (i + 1) / this.trail.length * 0.6;
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1.0;

    // Linha de Meta (Altura de corte superior)
    const goalLineY = pivotY - l1;
    ctx.strokeStyle = '#f59e0b';
    ctx.setLineDash([4, 4]);
    ctx.beginPath();
    ctx.moveTo(40, goalLineY);
    ctx.lineTo(w - 40, goalLineY);
    ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#f59e0b';
    ctx.font = '10px "JetBrains Mono", monospace';
    ctx.fillText('META (SWING ACIMA DA LINHA)', 45, goalLineY - 6);

    // Primeiro braço
    ctx.strokeStyle = '#38bdf8';
    ctx.lineWidth = 6;
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(pivotX, pivotY);
    ctx.lineTo(jointX, jointY);
    ctx.stroke();

    // Articulação intermediária
    ctx.fillStyle = '#f59e0b';
    ctx.beginPath();
    ctx.arc(jointX, jointY, 6, 0, Math.PI * 2);
    ctx.fill();

    // Segundo braço
    ctx.strokeStyle = '#a3e635';
    ctx.lineWidth = 5;
    ctx.beginPath();
    ctx.moveTo(jointX, jointY);
    ctx.lineTo(tipX, tipY);
    ctx.stroke();

    // Ponta final
    ctx.fillStyle = this.isGoalReached ? '#4ade80' : '#22c55e';
    ctx.beginPath();
    ctx.arc(tipX, tipY, 7, 0, Math.PI * 2);
    ctx.fill();

    // Pivô base
    ctx.fillStyle = '#cbd5e1';
    ctx.beginPath();
    ctx.arc(pivotX, pivotY, 8, 0, Math.PI * 2);
    ctx.fill();

    // Cabeçalho
    ctx.fillStyle = '#38bdf8';
    ctx.font = 'bold 12px "JetBrains Mono", monospace';
    ctx.fillText('⚡ SYSTEM 1 HUD | ACROBOT-V1', 20, 26);

    ctx.fillStyle = '#94a3b8';
    ctx.font = '11px "JetBrains Mono", monospace';
    ctx.fillText(`Passos Acrobáticos: ${this.steps}`, 20, 42);

    // Rodapé
    ctx.font = 'bold 11px "JetBrains Mono", monospace';
    if (this.isGoalReached) {
      ctx.fillStyle = '#4ade80';
      ctx.fillText('STATUS: META ATINGIDA 🏆 (SWING COMPLETO)', 20, h - 16);
    } else {
      ctx.fillStyle = '#e2e8f0';
      const torques = ['TORQUE ANTI-HORÁRIO ◀', 'SEM TORQUE (LIVRE)', 'TORQUE HORÁRIO ▶'];
      ctx.fillText(`AÇÃO: ${torques[this.lastAction] || 'IDLE'}`, 20, h - 16);
    }
  }
}

// =====================================================================
// AGENTE NEURAL SYSTEM 1 (ONNX RUNTIME WEB)
// =====================================================================

export interface DecisionResult {
  action: number;
  confidence: number;
  uncertainty: number;
  calibration: number;
  isUncertain: boolean;
  latencyMs: number;
  ranked?: { action: number; prob: number; logit: number }[];
  value?: number;
}

export class System1AgentWeb {
  session: any;
  ort: any;
  hxDim: number = 256;
  hx: Float32Array;
  prevObs: Float32Array | null = null;
  prevAction: number = 0;
  prevReward: number = 0.0;
  dynamicCalibration: number = 0.0;
  stagnationCount: number = 0;
  rewardEma: number = 0.0;
  actionHistory: number[] = [];

  constructor(session: any, ortInstance: any, initialHxDim?: number) {
    this.session = session;
    this.ort = ortInstance;
    let hxDim = initialHxDim || 256;
    try {
      if (session?.inputMetadata?.hx?.dimensions) {
        const dims = session.inputMetadata.hx.dimensions;
        const last = Number(dims[dims.length - 1]);
        if (last > 0) hxDim = last;
      }
    } catch (_) {}
    this.hxDim = hxDim;
    this.hx = new Float32Array(this.hxDim);
  }

  resetMemory() {
    this.hx.fill(0.0);
    this.prevObs = null;
    this.prevAction = 0;
    this.prevReward = 0.0;
    this.dynamicCalibration = 0.0;
    this.stagnationCount = 0;
    this.rewardEma = 0.0;
    this.actionHistory = [];
  }

  async actWithConfidence(
    obsArr: Float32Array,
    stepReward = 0.0,
    options: {
      avoidAction?: number | number[];
      temperature?: number;
      calibration?: number | 'auto';
      autoCalibrate?: boolean;
      deterministic?: boolean;
      qualityFilter?: boolean;
    } = {}
  ): Promise<DecisionResult> {
    const obsDim = obsArr.length;
    let deltaArr = new Float32Array(obsDim);
    if (this.prevObs) {
      for (let i = 0; i < obsDim; i++) deltaArr[i] = obsArr[i] - this.prevObs[i];
    }
    this.prevObs = new Float32Array(obsArr);
    this.prevReward = typeof stepReward === 'number' ? stepReward : 0.0;

    const t0 = performance.now();

    const tObs = new this.ort.Tensor('float32', obsArr, [1, 1, obsDim]);
    const tDelta = new this.ort.Tensor('float32', deltaArr, [1, 1, obsDim]);
    const tAct = new this.ort.Tensor('int64', BigInt64Array.from([BigInt(this.prevAction)]), [1, 1]);
    const tRew = new this.ort.Tensor('float32', new Float32Array([this.prevReward]), [1, 1, 1]);
    const tHx = new this.ort.Tensor('float32', this.hx, [1, 1, this.hxDim]);

    const feeds: Record<string, any> = {
      obs: tObs,
      delta_obs: tDelta,
      prev_action: tAct,
      prev_reward: tRew,
      hx: tHx,
    };

    let results: any;
    try {
      results = await this.session.run(feeds);
    } catch (runErr: any) {
      // Auto-adapta dinamicamente à dimensão de hx (ex: 512 vs 256)
      const msg = String(runErr?.message || runErr);
      const m = msg.match(/Expected:\s*(\d+)/i) || msg.match(/expected\s+\[\d+,\s*\d+,\s*(\d+)\]/i);
      if (m && Number(m[1]) > 0 && Number(m[1]) !== this.hxDim) {
        this.hxDim = Number(m[1]);
        this.hx = new Float32Array(this.hxDim);
        feeds.hx = new this.ort.Tensor('float32', this.hx, [1, 1, this.hxDim]);
        results = await this.session.run(feeds);
      } else {
        throw runErr;
      }
    }
    const latencyMs = performance.now() - t0;

    const logits = results.logits.data;
    if (results.next_hx && results.next_hx.data) {
      if (this.hx.length !== results.next_hx.data.length) {
        this.hxDim = results.next_hx.data.length;
        this.hx = new Float32Array(this.hxDim);
      }
      this.hx.set(results.next_hx.data);
    }

    // Softmax & Entropia de Shannon (Gating nos logits originais)
    let maxLogit = -Infinity;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > maxLogit) maxLogit = logits[i];
    }

    let sumExp = 0;
    const probs = new Float32Array(logits.length);
    for (let i = 0; i < logits.length; i++) {
      probs[i] = Math.exp(logits[i] - maxLogit);
      sumExp += probs[i];
    }

    let entropy = 0;
    let bestAction = 0;
    let maxProb = 0;
    const ranked: { action: number; prob: number; logit: number }[] = [];
    for (let i = 0; i < probs.length; i++) {
      probs[i] /= sumExp;
      ranked.push({ action: i, prob: probs[i], logit: logits[i] });
      if (probs[i] > maxProb) { maxProb = probs[i]; bestAction = i; }
      if (probs[i] > 1e-9) entropy -= probs[i] * Math.log(probs[i]);
    }
    ranked.sort((a, b) => b.prob - a.prob);

    const maxEntropy = Math.log(logits.length);
    const uncertainty = Math.min(1.0, Math.max(0.0, entropy / maxEntropy));

    // Calibração Contínua de Ação [0.0 = Determinístico/argmax, 1.0 = Estocástico Total] ou Auto-Calibração Homeostática
    let calib = 0.5;
    const isAuto = Boolean(options.autoCalibrate || options.calibration === 'auto');
    if (isAuto) {
      const r = this.prevReward;
      const deltaR = r - this.rewardEma;
      this.rewardEma = 0.9 * this.rewardEma + 0.1 * r;

      if (r > 0.01 || deltaR > 0.01) {
        this.stagnationCount = 0;
        this.dynamicCalibration = Math.max(0.0, this.dynamicCalibration - 0.20);
      } else {
        this.stagnationCount++;
        if (this.stagnationCount >= 2) {
          this.dynamicCalibration = Math.min(0.80, this.dynamicCalibration + 0.10);
        }
      }
      if (this.dynamicCalibration <= 0.0) {
        calib = 0.0;
      } else {
        calib = Math.min(0.80, Math.max(0.0, this.dynamicCalibration * (0.6 + 0.4 * uncertainty) + 0.10 * Math.max(0.0, uncertainty - 0.50)));
      }
    } else {
      const calibration = options.calibration !== undefined 
        ? options.calibration 
        : (options.temperature !== undefined ? options.temperature : (options.deterministic ? 0.0 : 0.5));
      calib = Math.min(1.0, Math.max(0.0, typeof calibration === 'number' ? calibration : 0.5));
    }

    // Resolução do conjunto de ações a evitar / quebra de ciclos e qualidade
    const avoidSet = new Set<number>();
    if (options.avoidAction !== undefined) {
      if (Array.isArray(options.avoidAction)) {
        options.avoidAction.forEach((a) => { if (typeof a === 'number' && a >= 0) avoidSet.add(a); });
      } else if (typeof options.avoidAction === 'number' && options.avoidAction >= 0) {
        avoidSet.add(options.avoidAction);
      }
    }

    const hist = this.actionHistory || [];
    if ((isAuto && this.stagnationCount >= 2) || (options.qualityFilter !== false && hist.length >= 3)) {
      if (hist.length >= 3 && hist[hist.length - 1] === hist[hist.length - 3]) {
        // Oscilação A -> B -> A sob estagnação: evita B para romper o ciclo vicioso
        avoidSet.add(hist[hist.length - 2]);
      } else if (hist.length >= 3 && hist[hist.length - 1] === hist[hist.length - 2] && hist[hist.length - 2] === hist[hist.length - 3]) {
        avoidSet.add(hist[hist.length - 1]);
      }
    }
    // Action Quality Filter & Inverse Pruning:
    // Em cubo atômico (12 ações), o movimento imediatamente inverso (ex: U -> U')
    // anula o passo anterior e desperdiça tempo. Evita reversão direta se houver histórico e maxProb < 0.96.
    if (options.qualityFilter !== false && hist.length >= 1 && logits.length === 12) {
      const last = hist[hist.length - 1];
      const inv = (last % 2 === 0) ? last + 1 : last - 1;
      if (maxProb < 0.96) {
        avoidSet.add(inv);
      }
    }

    let chosenAction = bestAction;
    if (calib > 0.01) {
      // Inovação proporcional com probabilidade = calib
      if (Math.random() < calib) {
        const T = 1.0 + 0.5 * calib;
        let maxScaled = -Infinity;
        for (let i = 0; i < logits.length; i++) {
          if (logits[i] / T > maxScaled) maxScaled = logits[i] / T;
        }
        let sumExpT = 0;
        const sampledProbs = new Float32Array(logits.length);
        for (let i = 0; i < logits.length; i++) {
          sampledProbs[i] = Math.exp((logits[i] / T) - maxScaled);
          sumExpT += sampledProbs[i];
        }
        for (let i = 0; i < logits.length; i++) sampledProbs[i] /= sumExpT;

        const randVal = Math.random();
        let cum = 0;
        for (let i = 0; i < sampledProbs.length; i++) {
          cum += sampledProbs[i];
          if (randVal <= cum || i === sampledProbs.length - 1) {
            chosenAction = i;
            break;
          }
        }
      } else {
        chosenAction = bestAction;
      }
    }

    // Quebra estrita de ciclos (Cycle Breaker):
    if (avoidSet.size > 0 && avoidSet.has(chosenAction) && ranked.length > 1) {
      const alt = ranked.find((item) => !avoidSet.has(item.action));
      chosenAction = alt ? alt.action : ranked[1].action;
    }

    if (!this.actionHistory) this.actionHistory = [];
    this.actionHistory.push(chosenAction);
    if (this.actionHistory.length > 8) this.actionHistory.shift();

    this.prevAction = chosenAction;

    return {
      action: chosenAction,
      confidence: maxProb,
      uncertainty,
      calibration: calib,
      isUncertain: uncertainty > 0.70 || maxProb < 0.50,
      latencyMs,
      ranked,
      value: results.value ? results.value.data[0] : 0.0,
    };
  }
}
