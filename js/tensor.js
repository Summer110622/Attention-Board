// Tiny numeric evaluator used for the live attention preview.
// It runs the block program on a scaled-down configuration with random weights.
// Tensors are { shape: number[], data: Float32Array } with the batch dimension dropped.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng) {
  const u = Math.max(rng(), 1e-12), v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const make = (shape, fill) => {
  const n = shape.reduce((a, b) => a * b, 1);
  const data = new Float32Array(n);
  if (fill) for (let i = 0; i < n; i++) data[i] = fill(i);
  return { shape, data };
};

// Scaled-down evaluation config derived from the symbolic one.
export function previewConfig(cfg) {
  const Dh = 4;
  let H = Math.max(1, Math.min(cfg.H, 4));
  let Hkv = Math.max(1, Math.min(cfg.Hkv, H));
  while (H % Hkv !== 0) Hkv -= 1;
  const T = Math.max(2, Math.min(cfg.T, 16));
  return { T, H, Hkv, Dh, D: H * Dh, scaleT: T / cfg.T };
}

function matmulRows(a, rows, inner, b, cols) {
  // a: rows x inner, b: inner x cols
  const out = new Float32Array(rows * cols);
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const av = a[i * inner + k];
      if (av === 0) continue;
      for (let j = 0; j < cols; j++) out[i * cols + j] += av * b[k * cols + j];
    }
  }
  return out;
}

function randomMatrix(rng, rows, cols) {
  const s = 1 / Math.sqrt(rows);
  return Float32Array.from({ length: rows * cols }, () => normal(rng) * s);
}

function softmaxRow(row, temp) {
  let m = -Infinity;
  for (let i = 0; i < row.length; i++) m = Math.max(m, row[i] / temp);
  let sum = 0;
  const out = new Float32Array(row.length);
  for (let i = 0; i < row.length; i++) {
    const e = row[i] === -Infinity ? 0 : Math.exp(row[i] / temp - m);
    out[i] = e; sum += e;
  }
  for (let i = 0; i < row.length; i++) out[i] = sum > 0 ? out[i] / sum : 0;
  return out;
}

function sparsemaxRow(row, temp) {
  const x = Array.from(row, (v) => (v === -Infinity ? -1e4 : v / temp));
  const z = x.slice().sort((a, b) => b - a);
  let cssv = 0, kz = 0, tauSum = 0;
  for (let k = 1; k <= z.length; k++) {
    cssv += z[k - 1];
    if (k * z[k - 1] > cssv - 1) { kz = k; tauSum = cssv; }
  }
  const tau = (tauSum - 1) / kz;
  return Float32Array.from(x, (v) => Math.max(v - tau, 0));
}

export function evaluate(program, analysis, seed = 7) {
  const pc = previewConfig(program.config);
  const rng = mulberry32(seed);
  const env = new Map();
  const outputs = {};
  let stopped = null;
  const { T, H, Hkv, Dh, D } = pc;

  for (const b of program.blocks) {
    if (stopped) break;
    const p = b.params;
    const info = analysis.perBlock[b.id];
    if (info.errors.length) { stopped = { blockId: b.id, key: 'eval.blockerror' }; break; }
    let out = null;
    const get = (n) => env.get(n);
    switch (b.type) {
      case 'input':
        out = make([T, D], () => normal(rng));
        break;
      case 'proj': {
        const x = get(p.src);
        const heads = p.heads === 'Hkv' ? Hkv : H;
        let y;
        if (p.kind === 'lowrank') {
          const r = Math.max(1, Math.min(p.rank, 4));
          const w1 = randomMatrix(rng, D, r), w2 = randomMatrix(rng, r, heads * Dh);
          y = matmulRows(matmulRows(x.data, T, D, w1, r), T, r, w2, heads * Dh);
        } else {
          y = matmulRows(x.data, T, D, randomMatrix(rng, D, heads * Dh), heads * Dh);
        }
        out = make([heads, T, Dh], (i) => {
          const h = Math.floor(i / (T * Dh)), t = Math.floor(i / Dh) % T, d = i % Dh;
          return y[t * heads * Dh + h * Dh + d];
        });
        break;
      }
      case 'mla': {
        const x = get(p.src);
        const nope = 4, rope = 2, vd = 4, dq = nope + rope;
        const rq = Math.max(1, Math.min(p.qRank || 4, 4)), rkv = Math.max(1, Math.min(p.kvRank, 4));
        const q = p.qRank > 0
          ? matmulRows(matmulRows(x.data, T, D, randomMatrix(rng, D, rq), rq), T, rq, randomMatrix(rng, rq, H * dq), H * dq)
          : matmulRows(x.data, T, D, randomMatrix(rng, D, H * dq), H * dq);
        const down = matmulRows(x.data, T, D, randomMatrix(rng, D, rkv + rope), rkv + rope);
        const ckv = new Float32Array(T * rkv), kpe = new Float32Array(T * rope);
        for (let t = 0; t < T; t++) {
          for (let j = 0; j < rkv; j++) ckv[t * rkv + j] = down[t * (rkv + rope) + j];
          for (let j = 0; j < rope; j++) kpe[t * rope + j] = down[t * (rkv + rope) + rkv + j];
        }
        const kv = matmulRows(ckv, T, rkv, randomMatrix(rng, rkv, H * (nope + vd)), H * (nope + vd));
        const rot = (arr, o, t) => {
          const half = rope / 2;
          for (let j = 0; j < half; j++) {
            const ang = t * Math.pow(p.base, -j / half), c = Math.cos(ang), s2 = Math.sin(ang);
            const x1 = arr[o + j], x2 = arr[o + half + j];
            arr[o + j] = x1 * c - x2 * s2; arr[o + half + j] = x2 * c + x1 * s2;
          }
        };
        for (let t = 0; t < T; t++) rot(kpe, t * rope, t);
        const qo = make([H, T, dq]), ko = make([H, T, dq]), vo = make([H, T, vd]);
        for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) {
          const dst = (h * T + t) * dq;
          for (let d = 0; d < dq; d++) qo.data[dst + d] = q[t * H * dq + h * dq + d];
          rot(qo.data, dst + nope, t);
          for (let d = 0; d < nope; d++) ko.data[dst + d] = kv[t * H * (nope + vd) + h * (nope + vd) + d];
          for (let d = 0; d < rope; d++) ko.data[dst + nope + d] = kpe[t * rope + d];
          for (let d = 0; d < vd; d++) vo.data[(h * T + t) * vd + d] = kv[t * H * (nope + vd) + h * (nope + vd) + nope + d];
        }
        env.set(p.outQ, qo); env.set(p.outK, ko); env.set(p.outV, vo);
        outputs[b.id] = qo;
        break;
      }
      case 'rope': {
        const x = get(p.target);
        const [h, , dh] = x.shape;
        const half = dh / 2;
        out = make(x.shape.slice());
        for (let a = 0; a < h; a++) for (let t = 0; t < T; t++) for (let j = 0; j < half; j++) {
          const ang = t * Math.pow(p.base, -j / half);
          const c = Math.cos(ang), s = Math.sin(ang);
          const o = (a * T + t) * dh;
          const x1 = x.data[o + j], x2 = x.data[o + half + j];
          out.data[o + j] = x1 * c - x2 * s;
          out.data[o + half + j] = x2 * c + x1 * s;
        }
        break;
      }
      case 'score': {
        const q = get(p.q), k = get(p.k);
        const hk = k.shape[0];
        const rep = H / hk;
        out = make([H, T, T]);
        const dq = q.shape[2];
        const norm = (arr, o) => { let n = 0; for (let d = 0; d < dq; d++) n += arr[o + d] * arr[o + d]; return Math.sqrt(n) || 1; };
        for (let h = 0; h < H; h++) {
          const kh = Math.floor(h / rep);
          for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
            let s = 0;
            const qo = (h * T + i) * dq, ko = (kh * T + j) * dq;
            for (let d = 0; d < dq; d++) s += q.data[qo + d] * k.data[ko + d];
            if (p.kind === 'cosine') s /= norm(q.data, qo) * norm(k.data, ko);
            if (p.scale === 'sqrt') s /= Math.sqrt(dq);
            out.data[(h * T + i) * T + j] = s;
          }
        }
        break;
      }
      case 'mask': {
        const s = get(p.target);
        out = make(s.shape.slice(), (i) => s.data[i]);
        const win = Math.max(1, Math.round(p.window * pc.scaleT));
        const kk = Math.max(1, Math.min(p.k, T));
        for (let h = 0; h < H; h++) for (let i = 0; i < T; i++) {
          const o = (h * T + i) * T;
          if (p.kind === 'topk') {
            const row = Array.from(out.data.slice(o, o + T)).sort((a, b) => b - a);
            const thr = row[Math.min(kk, T) - 1];
            for (let j = 0; j < T; j++) if (out.data[o + j] < thr) out.data[o + j] = -Infinity;
          } else {
            for (let j = 0; j < T; j++) {
              const masked = p.kind === 'causal' ? j > i : (j > i || i - j >= win);
              if (masked) out.data[o + j] = -Infinity;
            }
          }
        }
        break;
      }
      case 'norm': {
        const s = get(p.target);
        out = make(s.shape.slice());
        for (let r = 0; r < H * T; r++) {
          const row = s.data.slice(r * T, (r + 1) * T);
          let res;
          if (p.fn === 'softmax') res = softmaxRow(row, p.temp);
          else if (p.fn === 'sparsemax') res = sparsemaxRow(row, p.temp);
          else if (p.fn === 'relu2') {
            res = Float32Array.from(row, (v) => (v === -Infinity ? 0 : Math.pow(Math.max(v / p.temp, 0), 2)));
            const sum = res.reduce((a, b) => a + b, 0) + 1e-6;
            res = res.map((v) => v / sum);
          } else res = Float32Array.from(row, (v) => (v === -Infinity ? 0 : 1 / (1 + Math.exp(-v / p.temp))));
          out.data.set(res, r * T);
        }
        break;
      }
      case 'dropout':
        out = get(p.target);
        break;
      case 'mix': {
        const a = get(p.a), v = get(p.v);
        const rep = H / v.shape[0];
        const dv = v.shape[2];
        out = make([H, T, dv]);
        for (let h = 0; h < H; h++) {
          const vh = Math.floor(h / rep);
          for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
            const w = a.data[(h * T + i) * T + j];
            if (w === 0) continue;
            for (let d = 0; d < dv; d++) out.data[(h * T + i) * dv + d] += w * v.data[(vh * T + j) * dv + d];
          }
        }
        break;
      }
      case 'outproj': {
        const o = get(p.src);
        const dv = o.shape[2], inner = H * dv;
        const cat = new Float32Array(T * inner);
        for (let h = 0; h < H; h++) for (let t = 0; t < T; t++) for (let d = 0; d < dv; d++) cat[t * inner + h * dv + d] = o.data[(h * T + t) * dv + d];
        out = { shape: [T, D], data: matmulRows(cat, T, inner, randomMatrix(rng, inner, D), D) };
        break;
      }
      case 'ret':
        break;
      case 'custom':
        stopped = { blockId: b.id, key: 'eval.custom' };
        break;
      default: break;
    }
    if (out) {
      const name = info.out ? info.out.name : null;
      if (name) env.set(name, out);
      outputs[b.id] = out;
    }
  }
  return { cfg: pc, outputs, stopped };
}

// Statistics for an attention-weight tensor of shape [H, T, T].
export function attentionStats(t) {
  const [H, T] = t.shape;
  let ent = 0, sparse = 0, maxw = 0;
  const rows = H * T;
  for (let r = 0; r < rows; r++) {
    let e = 0, rowMax = 0;
    for (let j = 0; j < T; j++) {
      const w = t.data[r * T + j];
      if (w > 1e-9) e -= w * Math.log(w);
      if (w < 1e-6) sparse++;
      rowMax = Math.max(rowMax, w);
    }
    ent += e; maxw += rowMax;
  }
  return { entropy: ent / rows, sparsity: sparse / (rows * T), meanMax: maxw / rows };
}

export const isAttentionMap = (t) => t && t.shape.length === 3 && t.shape[1] === t.shape[2];
