// Block registry, symbolic shape checking and PyTorch code generation.
// Pure logic: no DOM access, so it can be unit-tested with Node.

// Dh = 0 means "derive from D / H". Set it explicitly for models whose head_dim * H != hidden_size.
export const DEFAULT_CONFIG = { B: 2, T: 128, D: 512, H: 8, Hkv: 8, Dh: 0 };

export const effDh = (cfg) => (cfg.Dh > 0 ? cfg.Dh : cfg.D / cfg.H);

export const CATEGORIES = {
  input:   { color: '#2F6FDE', tint: '#E8F0FD' },
  project: { color: '#0E8F86', tint: '#E1F4F2' },
  score:   { color: '#C8611A', tint: '#FCEADB' },
  mask:    { color: '#7A4FD0', tint: '#EEE8FA' },
  norm:    { color: '#C2377A', tint: '#FBE5EF' },
  mix:     { color: '#3B8C2F', tint: '#E4F3DF' },
  custom:  { color: '#5B6472', tint: '#E9ECF0' }
};

const NAME_RE = /^[A-Za-z_][A-Za-z0-9_]*$/;
const fmtShape = (s) => '[' + s.join(', ') + ']';
export { fmtShape };

export function dimValue(sym, cfg) {
  if (typeof sym === 'number') return sym;
  if (sym === 'Dh') return effDh(cfg);
  return cfg[sym];
}

const headsSym = (p) => (p.heads === 'Hkv' ? 'Hkv' : 'H');

// Each definition: cat, defaults(), fields (for the inspector), parts(p) (block label),
// check(p, ctx) -> { out, params, cost } and pushes errors through ctx.err().
export const BLOCKS = {
  input: {
    cat: 'input',
    defaults: () => ({ name: 'x' }),
    fields: [{ key: 'name', kind: 'text' }],
    parts: (p) => [['when', 0], ['attention(' + p.name + ')', 1], ['is called', 0]],
    check(p, ctx) {
      ctx.name(p.name);
      return { out: { name: p.name, shape: ['B', 'T', 'D'] }, cost: '—' };
    }
  },
  proj: {
    cat: 'project',
    defaults: () => ({ src: 'x', out: 'q', kind: 'linear', rank: 64, heads: 'H' }),
    fields: [
      { key: 'src', kind: 'tensor' },
      { key: 'out', kind: 'text' },
      { key: 'kind', kind: 'select', options: ['linear', 'lowrank'] },
      { key: 'rank', kind: 'number', min: 1, step: 1, when: (p) => p.kind === 'lowrank' },
      { key: 'heads', kind: 'select', options: ['H', 'Hkv'] }
    ],
    parts: (p) => [[p.out.toUpperCase() + ' =', 0], [p.src, 1], ['·', 0], ['W_' + p.out, 1]],
    check(p, ctx) {
      const s = ctx.need(p.src);
      ctx.name(p.out);
      if (s && !(s.length === 3 && s[2] === 'D')) ctx.err('err.shape', { expect: '[B, T, D]', got: fmtShape(s) });
      const hs = headsSym(p);
      const { D, H, Hkv } = ctx.cfg;
      if (hs === 'Hkv' && H % Hkv !== 0) ctx.err('err.gqa', { H, Hkv });
      const heads = dimValue(hs, ctx.cfg);
      const dh = effDh(ctx.cfg);
      const params = p.kind === 'lowrank' ? D * p.rank + p.rank * heads * dh : D * heads * dh;
      return { out: { name: p.out, shape: ['B', hs, 'T', 'Dh'] }, params, cost: 'O(B·T·D·' + hs + '·Dh)' };
    }
  },
  rope: {
    cat: 'project',
    defaults: () => ({ target: 'q', base: 10000 }),
    fields: [{ key: 'target', kind: 'tensor' }, { key: 'base', kind: 'number', min: 1 }],
    parts: (p) => [['rotate', 0], [p.target, 1], ['(RoPE)', 0]],
    check(p, ctx) {
      const s = ctx.need(p.target);
      if (s && s.length !== 4) ctx.err('err.shape', { expect: '[B, H, T, Dh]', got: fmtShape(s) });
      if (s && s.length === 4 && dimValue(s[3], ctx.cfg) % 2 !== 0) ctx.err('err.rope_even', {});
      return { out: s ? { name: p.target, shape: s } : null, cost: 'O(B·H·T·Dh)' };
    }
  },
  score: {
    cat: 'score',
    defaults: () => ({ q: 'q', k: 'k', out: 's', kind: 'dot', scale: 'sqrt' }),
    fields: [
      { key: 'q', kind: 'tensor' }, { key: 'k', kind: 'tensor' }, { key: 'out', kind: 'text' },
      { key: 'kind', kind: 'select', options: ['dot', 'cosine'] },
      { key: 'scale', kind: 'select', options: ['sqrt', 'none'] }
    ],
    parts: (p) => [[p.out.toUpperCase() + ' =', 0], [p.q, 1], ['·', 0], [p.k + 'ᵀ', 1]],
    check(p, ctx) {
      const q = ctx.need(p.q), k = ctx.need(p.k);
      ctx.name(p.out);
      if (q && !(q.length === 4 && q[1] === 'H')) ctx.err('err.shape', { expect: '[B, H, T, Dh]', got: fmtShape(q) });
      if (k && !(k.length === 4 && (k[1] === 'H' || k[1] === 'Hkv'))) ctx.err('err.shape', { expect: '[B, H|Hkv, T, Dh]', got: fmtShape(k) });
      if (q && k && q.length === 4 && k.length === 4 && q[3] !== k[3]) ctx.err('err.dim_mismatch', { a: q[3], b: k[3] });
      return { out: { name: p.out, shape: ['B', 'H', 'T', 'T'] }, cost: 'O(B·H·T²·Dh)' };
    }
  },
  mask: {
    cat: 'mask',
    defaults: () => ({ target: 's', kind: 'causal', window: 32, k: 16 }),
    fields: [
      { key: 'target', kind: 'tensor' },
      { key: 'kind', kind: 'select', options: ['causal', 'sliding', 'topk'] },
      { key: 'window', kind: 'number', min: 1, step: 1, when: (p) => p.kind === 'sliding' },
      { key: 'k', kind: 'number', min: 1, step: 1, when: (p) => p.kind === 'topk' }
    ],
    parts: (p) => [[p.target + ' +=', 0], ['mask', 1], ['(', 0], [p.kind, 1], [')', 0]],
    check(p, ctx) {
      const s = ctx.need(p.target);
      if (s && !(s.length === 4 && s[2] === 'T' && s[3] === 'T')) ctx.err('err.shape', { expect: '[B, H, T, T]', got: fmtShape(s) });
      if (p.kind === 'sliding' && !(p.window >= 1)) ctx.err('err.positive', { field: 'window' });
      if (p.kind === 'topk' && !(p.k >= 1)) ctx.err('err.positive', { field: 'k' });
      if (p.kind === 'topk' && p.k > ctx.cfg.T) ctx.err('err.topk_gt_t', { k: p.k, T: ctx.cfg.T });
      return { out: s ? { name: p.target, shape: s } : null, cost: 'O(B·H·T²)' };
    }
  },
  norm: {
    cat: 'norm',
    defaults: () => ({ target: 's', out: 'a', fn: 'softmax', temp: 1 }),
    fields: [
      { key: 'target', kind: 'tensor' }, { key: 'out', kind: 'text' },
      { key: 'fn', kind: 'select', options: ['softmax', 'sparsemax', 'relu2', 'sigmoid'] },
      { key: 'temp', kind: 'number', min: 0.0001, step: 0.1 }
    ],
    alts: ['softmax', 'sparsemax', 'relu2', 'sigmoid'],
    parts: (p) => [[p.out.toUpperCase() + ' =', 0], [p.fn, 1], ['(' + p.target + ')', 0]],
    check(p, ctx) {
      const s = ctx.need(p.target);
      ctx.name(p.out);
      if (s && !(s.length === 4 && s[3] === 'T')) ctx.err('err.shape', { expect: '[B, H, T, T]', got: fmtShape(s) });
      if (!(p.temp > 0)) ctx.err('err.positive', { field: 'temp' });
      return { out: s ? { name: p.out, shape: s } : null, cost: 'O(B·H·T²)' };
    }
  },
  dropout: {
    cat: 'norm',
    defaults: () => ({ target: 'a', p: 0.1 }),
    fields: [{ key: 'target', kind: 'tensor' }, { key: 'p', kind: 'number', min: 0, max: 0.99, step: 0.05 }],
    parts: (p) => [['dropout', 0], [p.target, 1], ['p=' + p.p, 0]],
    check(p, ctx) {
      const s = ctx.need(p.target);
      if (!(p.p >= 0 && p.p < 1)) ctx.err('err.prob', {});
      return { out: s ? { name: p.target, shape: s } : null, cost: 'O(B·H·T²)' };
    }
  },
  mix: {
    cat: 'mix',
    defaults: () => ({ a: 'a', v: 'v', out: 'o' }),
    fields: [{ key: 'a', kind: 'tensor' }, { key: 'v', kind: 'tensor' }, { key: 'out', kind: 'text' }],
    parts: (p) => [[p.out.toUpperCase() + ' =', 0], [p.a, 1], ['·', 0], [p.v, 1]],
    check(p, ctx) {
      const a = ctx.need(p.a), v = ctx.need(p.v);
      ctx.name(p.out);
      if (a && !(a.length === 4 && a[2] === 'T' && a[3] === 'T')) ctx.err('err.shape', { expect: '[B, H, T, T]', got: fmtShape(a) });
      if (v && !(v.length === 4 && (v[1] === 'H' || v[1] === 'Hkv') && v[2] === 'T')) ctx.err('err.shape', { expect: '[B, H|Hkv, T, Dh]', got: fmtShape(v) });
      return { out: { name: p.out, shape: ['B', 'H', 'T', v && v.length === 4 ? v[3] : 'Dh'] }, cost: 'O(B·H·T²·Dh)' };
    }
  },
  mla: {
    cat: 'project',
    defaults: () => ({ src: 'x', outQ: 'q', outK: 'k', outV: 'v', qRank: 1536, kvRank: 512, nope: 128, rope: 64, vDim: 128, base: 10000 }),
    fields: [
      { key: 'src', kind: 'tensor' },
      { key: 'outQ', kind: 'text' }, { key: 'outK', kind: 'text' }, { key: 'outV', kind: 'text' },
      { key: 'qRank', kind: 'number', min: 0, step: 1 }, { key: 'kvRank', kind: 'number', min: 1, step: 1 },
      { key: 'nope', kind: 'number', min: 1, step: 1 }, { key: 'rope', kind: 'number', min: 2, step: 2 },
      { key: 'vDim', kind: 'number', min: 1, step: 1 }, { key: 'base', kind: 'number', min: 1 }
    ],
    parts: (p) => [[p.outQ.toUpperCase() + ', ' + p.outK.toUpperCase() + ', ' + p.outV.toUpperCase() + ' =', 0], ['MLA', 1], ['(', 0], [p.src, 1], [')', 0]],
    check(p, ctx) {
      const s = ctx.need(p.src);
      [p.outQ, p.outK, p.outV].forEach((n) => ctx.name(n));
      if (s && !(s.length === 3 && s[2] === 'D')) ctx.err('err.shape', { expect: '[B, T, D]', got: fmtShape(s) });
      ['kvRank', 'nope', 'rope', 'vDim'].forEach((f) => { if (!(p[f] >= 1)) ctx.err('err.positive', { field: f }); });
      if (!(p.qRank >= 0)) ctx.err('err.positive', { field: 'qRank' });
      if (p.rope % 2 !== 0) ctx.err('err.rope_even', {});
      const { D, H } = ctx.cfg;
      const dq = p.nope + p.rope;
      const qParams = p.qRank > 0 ? D * p.qRank + p.qRank * H * dq : D * H * dq;
      const kvParams = D * (p.kvRank + p.rope) + p.kvRank * H * (p.nope + p.vDim);
      return {
        outs: [
          { name: p.outQ, shape: ['B', 'H', 'T', dq] },
          { name: p.outK, shape: ['B', 'H', 'T', dq] },
          { name: p.outV, shape: ['B', 'H', 'T', p.vDim] }
        ],
        params: qParams + kvParams, cost: 'O(B·T·D·(r_q+r_kv))'
      };
    }
  },
  outproj: {
    cat: 'project',
    defaults: () => ({ src: 'o', out: 'y' }),
    fields: [{ key: 'src', kind: 'tensor' }, { key: 'out', kind: 'text' }],
    parts: (p) => [[p.out.toUpperCase() + ' = concat(' + p.src + ') ·', 0], ['W_o', 1]],
    check(p, ctx) {
      const o = ctx.need(p.src);
      ctx.name(p.out);
      if (o && !(o.length === 4 && o[1] === 'H')) ctx.err('err.shape', { expect: '[B, H, T, Dh]', got: fmtShape(o) });
      const inner = ctx.cfg.H * (o && o.length === 4 ? dimValue(o[3], ctx.cfg) : effDh(ctx.cfg));
      return { out: { name: p.out, shape: ['B', 'T', 'D'] }, params: inner * ctx.cfg.D, cost: 'O(B·T·H·d·D)' };
    }
  },
  ret: {
    cat: 'input',
    defaults: () => ({ src: 'y' }),
    fields: [{ key: 'src', kind: 'tensor' }],
    parts: (p) => [['return', 0], [p.src, 1]],
    check(p, ctx) {
      ctx.need(p.src);
      return { out: null, cost: '—' };
    }
  },
  custom: {
    cat: 'custom',
    defaults: () => ({ title: 'Custom op', inputs: 's', outName: 's', outShape: 'B, H, T, T', code: 's = s * 1.0' }),
    fields: [
      { key: 'title', kind: 'text' },
      { key: 'inputs', kind: 'text' },
      { key: 'outName', kind: 'text' },
      { key: 'outShape', kind: 'text' },
      { key: 'code', kind: 'textarea' }
    ],
    parts: (p) => [[p.title, 1], ['(' + p.inputs + ')', 0]],
    check(p, ctx) {
      const ins = String(p.inputs || '').split(',').map((s) => s.trim()).filter(Boolean);
      ins.forEach((n) => ctx.need(n));
      let out = null;
      if (p.outName) {
        ctx.name(p.outName);
        const shape = String(p.outShape || '').replace(/[\[\]]/g, '').split(',').map((s) => s.trim()).filter(Boolean)
          .map((s) => (/^\d+$/.test(s) ? Number(s) : s));
        if (!shape.length) ctx.err('err.noshape', {});
        else out = { name: p.outName, shape };
      }
      return { out, cost: '?' };
    }
  }
};

export const PALETTE = [
  { cat: 'input', types: ['input', 'ret'] },
  { cat: 'project', types: ['proj', 'mla', 'rope', 'outproj'] },
  { cat: 'score', types: ['score'] },
  { cat: 'mask', types: ['mask'] },
  { cat: 'norm', types: ['norm', 'dropout'] },
  { cat: 'mix', types: ['mix'] },
  { cat: 'custom', types: ['custom'] }
];

let idCounter = 0;
export function newBlock(type, overrides = {}) {
  idCounter += 1;
  return { id: 'b' + Date.now().toString(36) + idCounter, type, params: Object.assign(BLOCKS[type].defaults(), overrides) };
}

export function makeProgram(name, config, blocks) {
  return { version: 1, name, config: Object.assign({}, DEFAULT_CONFIG, config), blocks };
}

// ---------------------------------------------------------------------------
// Analysis
// ---------------------------------------------------------------------------
export function analyze(program) {
  const cfg = program.config;
  const env = new Map();
  const perBlock = {};
  const errors = [];
  const tensors = [];
  let params = 0;

  ['B', 'T', 'D', 'H', 'Hkv'].forEach((k) => {
    if (!(Number.isInteger(cfg[k]) && cfg[k] >= 1)) errors.push({ blockId: null, key: 'err.cfg_positive', args: { field: k } });
  });
  if (!(cfg.Dh === undefined || (Number.isInteger(cfg.Dh) && cfg.Dh >= 0))) errors.push({ blockId: null, key: 'err.cfg_positive', args: { field: 'Dh' } });
  if (!(cfg.Dh > 0) && Number.isInteger(cfg.D) && Number.isInteger(cfg.H) && cfg.H >= 1 && cfg.D % cfg.H !== 0) errors.push({ blockId: null, key: 'err.cfg_dh', args: { D: cfg.D, H: cfg.H } });
  if (Number.isInteger(cfg.H) && Number.isInteger(cfg.Hkv) && cfg.Hkv >= 1 && cfg.H % cfg.Hkv !== 0) errors.push({ blockId: null, key: 'err.gqa', args: { H: cfg.H, Hkv: cfg.Hkv } });

  program.blocks.forEach((b) => {
    const def = BLOCKS[b.type];
    const info = { ins: [], out: null, outs: [], errors: [], cost: '—', params: 0, scope: Array.from(env.keys()) };
    const ctx = {
      cfg,
      need(name) {
        const s = env.get(name);
        if (!s) { info.errors.push({ key: 'err.undefined', args: { name } }); return null; }
        info.ins.push({ name, shape: s });
        return s;
      },
      err(key, args) { info.errors.push({ key, args }); },
      name(n) { if (!NAME_RE.test(String(n || ''))) info.errors.push({ key: 'err.badname', args: { name: n } }); }
    };
    const r = def.check(b.params, ctx) || {};
    const outs = r.outs || (r.out ? [r.out] : []);
    info.outs = outs;
    info.out = outs[0] || null;
    info.cost = r.cost || '—';
    info.params = r.params || 0;
    params += info.params;
    outs.forEach((o) => { env.set(o.name, o.shape); tensors.push({ name: o.name, shape: o.shape, blockId: b.id }); });
    info.errors.forEach((e) => errors.push({ blockId: b.id, key: e.key, args: e.args }));
    perBlock[b.id] = info;
  });

  const warnings = [];
  if (!program.blocks.some((b) => b.type === 'ret')) warnings.push({ key: 'warn.noreturn', args: {} });
  return {
    perBlock, errors, warnings, tensors, params,
    attnEntries: cfg.B * cfg.H * cfg.T * cfg.T,
    ok: errors.length === 0
  };
}

// ---------------------------------------------------------------------------
// PyTorch code generation
// ---------------------------------------------------------------------------
const HELPERS = {
  repeat_kv: [
    'def repeat_kv(x, n_rep):',
    '    # [B, Hkv, T, Dh] -> [B, Hkv * n_rep, T, Dh]',
    '    if n_rep == 1:',
    '        return x',
    '    B, Hkv, T, Dh = x.shape',
    '    return x[:, :, None].expand(B, Hkv, n_rep, T, Dh).reshape(B, Hkv * n_rep, T, Dh)'
  ],
  rope: [
    'def apply_rope(x, base=10000.0):',
    '    # x: [B, H, T, Dh] (rotate-half convention)',
    '    T, Dh = x.shape[-2], x.shape[-1]',
    '    half = Dh // 2',
    '    inv_freq = base ** (-torch.arange(0, half, device=x.device, dtype=torch.float32) / half)',
    '    angles = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * inv_freq[None, :]',
    '    cos, sin = angles.cos().to(x.dtype), angles.sin().to(x.dtype)',
    '    x1, x2 = x[..., :half], x[..., half:]',
    '    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)'
  ],
  rmsnorm: [
    'class RMSNorm(nn.Module):',
    '    def __init__(self, dim, eps=1e-6):',
    '        super().__init__()',
    '        self.weight = nn.Parameter(torch.ones(dim))',
    '        self.eps = eps',
    '',
    '    def forward(self, x):',
    '        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight'
  ],
  causal_mask: [
    'def causal_mask(T, device):',
    '    return torch.ones(T, T, dtype=torch.bool, device=device).triu(1)'
  ],
  sliding_mask: [
    'def sliding_mask(T, window, device):',
    '    i = torch.arange(T, device=device)[:, None]',
    '    j = torch.arange(T, device=device)[None, :]',
    '    return (j > i) | (i - j >= window)'
  ],
  sparsemax: [
    'def sparsemax(x, dim=-1):',
    '    # Martins & Astudillo (2016). -inf is clamped so masked entries get exactly 0.',
    '    x = x.masked_fill(x == float("-inf"), -1e4)',
    '    z, _ = torch.sort(x, dim=dim, descending=True)',
    '    k = torch.arange(1, x.size(dim) + 1, device=x.device, dtype=x.dtype)',
    '    shape = [1] * x.dim(); shape[dim] = -1',
    '    k = k.view(shape)',
    '    cssv = z.cumsum(dim) - 1',
    '    support = (k * z) > cssv',
    '    k_z = support.sum(dim=dim, keepdim=True)',
    '    tau = cssv.gather(dim, k_z - 1) / k_z.to(x.dtype)',
    '    return torch.clamp(x - tau, min=0)'
  ]
};
const HELPER_ORDER = ['repeat_kv', 'rope', 'rmsnorm', 'causal_mask', 'sliding_mask', 'sparsemax'];

export function generateCode(program, analysis) {
  const cfg = program.config;
  const helpers = new Set();
  const init = [];
  const fwd = [];
  const used = new Set();
  const attr = (base, id) => {
    let n = base;
    if (used.has(n)) n = base + '_' + id.slice(-3);
    used.add(n);
    return n;
  };
  const shapeC = (info) => (info && info.out ? '  # ' + fmtShape(info.out.shape) : '');
  const hExpr = (sym) => (sym === 'Hkv' ? 'Hkv' : 'H');

  program.blocks.forEach((b) => {
    const p = b.params;
    const info = analysis.perBlock[b.id];
    const add = (t) => fwd.push({ blockId: b.id, text: t });
    const addInit = (t) => init.push({ blockId: b.id, text: t });
    switch (b.type) {
      case 'input':
        if (p.name !== 'x') add(p.name + ' = x');
        else add('# x: [B, T, D]');
        break;
      case 'proj': {
        const a = attr('W_' + p.out, b.id);
        const hs = headsSym(p);
        const hArg = hs === 'Hkv' ? 'n_kv_heads' : 'n_heads';
        if (p.kind === 'lowrank') {
          addInit('self.' + a + ' = nn.Sequential(nn.Linear(d_model, ' + p.rank + ', bias=False), nn.Linear(' + p.rank + ', ' + hArg + ' * self.Dh, bias=False))');
        } else {
          addInit('self.' + a + ' = nn.Linear(d_model, ' + hArg + ' * self.Dh, bias=False)');
        }
        add(p.out + ' = self.' + a + '(' + p.src + ').view(B, T, ' + hExpr(hs) + ', Dh).transpose(1, 2)' + shapeC(info));
        break;
      }
      case 'mla': {
        helpers.add('rope'); helpers.add('rmsnorm');
        const id = b.id.slice(-3);
        const dq = p.nope + p.rope;
        if (p.qRank > 0) {
          addInit('self.W_q_down_' + id + ' = nn.Linear(d_model, ' + p.qRank + ', bias=False)');
          addInit('self.q_norm_' + id + ' = RMSNorm(' + p.qRank + ')');
          addInit('self.W_q_up_' + id + ' = nn.Linear(' + p.qRank + ', n_heads * ' + dq + ', bias=False)');
        } else {
          addInit('self.W_q_' + id + ' = nn.Linear(d_model, n_heads * ' + dq + ', bias=False)');
        }
        addInit('self.W_kv_down_' + id + ' = nn.Linear(d_model, ' + p.kvRank + ' + ' + p.rope + ', bias=False)  # latent c_kv + shared RoPE key');
        addInit('self.kv_norm_' + id + ' = RMSNorm(' + p.kvRank + ')');
        addInit('self.W_kv_up_' + id + ' = nn.Linear(' + p.kvRank + ', n_heads * (' + p.nope + ' + ' + p.vDim + '), bias=False)');
        add('# Multi-head Latent Attention: low-rank q / kv, decoupled RoPE');
        const qsrc = p.qRank > 0 ? 'self.W_q_up_' + id + '(self.q_norm_' + id + '(self.W_q_down_' + id + '(' + p.src + ')))' : 'self.W_q_' + id + '(' + p.src + ')';
        add(p.outQ + ' = ' + qsrc + '.view(B, T, H, ' + dq + ').transpose(1, 2)');
        add('q_nope, q_pe = ' + p.outQ + '.split([' + p.nope + ', ' + p.rope + '], dim=-1)');
        add('c_kv, k_pe = self.W_kv_down_' + id + '(' + p.src + ').split([' + p.kvRank + ', ' + p.rope + '], dim=-1)');
        add('kv = self.W_kv_up_' + id + '(self.kv_norm_' + id + '(c_kv)).view(B, T, H, ' + p.nope + ' + ' + p.vDim + ').transpose(1, 2)');
        add('k_nope, ' + p.outV + ' = kv.split([' + p.nope + ', ' + p.vDim + '], dim=-1)');
        add('q_pe = apply_rope(q_pe, base=' + Number(p.base).toFixed(1) + ')');
        add('k_pe = apply_rope(k_pe.unsqueeze(1), base=' + Number(p.base).toFixed(1) + ')  # shared across heads');
        add(p.outQ + ' = torch.cat([q_nope, q_pe], dim=-1)  # [B, H, T, ' + dq + ']');
        add(p.outK + ' = torch.cat([k_nope, k_pe.expand(-1, H, -1, -1)], dim=-1)  # [B, H, T, ' + dq + ']');
        break;
      }
      case 'rope':
        helpers.add('rope');
        add(p.target + ' = apply_rope(' + p.target + ', base=' + Number(p.base).toFixed(1) + ')');
        break;
      case 'score': {
        const kShape = (info.ins[1] && info.ins[1].shape) || [];
        let kName = p.k;
        if (kShape[1] === 'Hkv') {
          helpers.add('repeat_kv');
          kName = p.k + '_rep';
          add(kName + ' = repeat_kv(' + p.k + ', H // Hkv)');
        }
        const qd = info.ins[0] && info.ins[0].shape ? info.ins[0].shape[3] : 'Dh';
        const scale = p.scale === 'sqrt' ? ' / math.sqrt(' + (typeof qd === 'number' ? qd : 'Dh') + ')' : '';
        if (p.kind === 'cosine') {
          add(p.out + ' = (F.normalize(' + p.q + ', dim=-1) @ F.normalize(' + kName + ', dim=-1).transpose(-2, -1))' + scale + shapeC(info));
        } else {
          add(p.out + ' = (' + p.q + ' @ ' + kName + '.transpose(-2, -1))' + scale + shapeC(info));
        }
        break;
      }
      case 'mask':
        if (p.kind === 'causal') {
          helpers.add('causal_mask');
          add(p.target + ' = ' + p.target + '.masked_fill(causal_mask(T, ' + p.target + '.device), float("-inf"))');
        } else if (p.kind === 'sliding') {
          helpers.add('sliding_mask');
          add(p.target + ' = ' + p.target + '.masked_fill(sliding_mask(T, ' + p.window + ', ' + p.target + '.device), float("-inf"))');
        } else {
          add('_thr = ' + p.target + '.topk(' + p.k + ', dim=-1).values[..., -1:]');
          add(p.target + ' = ' + p.target + '.masked_fill(' + p.target + ' < _thr, float("-inf"))');
        }
        break;
      case 'norm': {
        const tau = Number(p.temp) === 1 ? '' : ' / ' + Number(p.temp);
        if (p.fn === 'softmax') add(p.out + ' = torch.softmax(' + p.target + tau + ', dim=-1)' + shapeC(info));
        else if (p.fn === 'sparsemax') { helpers.add('sparsemax'); add(p.out + ' = sparsemax(' + p.target + tau + ', dim=-1)' + shapeC(info)); }
        else if (p.fn === 'relu2') {
          add(p.out + ' = F.relu(' + p.target + tau + ') ** 2');
          add(p.out + ' = ' + p.out + ' / (' + p.out + '.sum(-1, keepdim=True) + 1e-6)' + shapeC(info));
        } else add(p.out + ' = torch.sigmoid(' + p.target + tau + ')' + shapeC(info));
        break;
      }
      case 'dropout': {
        const a = attr('drop_' + b.id.slice(-3), b.id);
        addInit('self.' + a + ' = nn.Dropout(' + p.p + ')');
        add(p.target + ' = self.' + a + '(' + p.target + ')');
        break;
      }
      case 'mix': {
        const vShape = (info.ins[1] && info.ins[1].shape) || [];
        let vName = p.v;
        if (vShape[1] === 'Hkv') {
          helpers.add('repeat_kv');
          vName = p.v + '_rep';
          add(vName + ' = repeat_kv(' + p.v + ', H // Hkv)');
        }
        add(p.out + ' = ' + p.a + ' @ ' + vName + shapeC(info));
        break;
      }
      case 'outproj': {
        const a = attr('W_o', b.id);
        const vd = info.ins[0] && info.ins[0].shape ? info.ins[0].shape[3] : 'Dh';
        const inner = typeof vd === 'number' ? 'n_heads * ' + vd : 'n_heads * self.Dh';
        addInit('self.' + a + ' = nn.Linear(' + inner + ', d_model, bias=False)');
        add(p.out + ' = self.' + a + '(' + p.src + '.transpose(1, 2).reshape(B, T, H * ' + (typeof vd === 'number' ? vd : 'Dh') + '))' + shapeC(info));
        break;
      }
      case 'ret':
        add('return ' + p.src);
        break;
      case 'custom':
        add('# ' + p.title);
        String(p.code || '').split('\n').forEach((ln) => add(ln));
        break;
      default: break;
    }
  });

  const lines = [];
  const push = (text, blockId = null) => lines.push({ text, blockId });
  ['import math', 'import torch', 'import torch.nn as nn', 'import torch.nn.functional as F', ''].forEach((t) => push(t));
  HELPER_ORDER.filter((h) => helpers.has(h)).forEach((h) => {
    HELPERS[h].forEach((t) => push(t));
    push(''); push('');
  });
  if (!helpers.size) push('');
  push('class CustomAttention(nn.Module):');
  const explicitDh = cfg.Dh > 0;
  push('    def __init__(self, d_model=' + cfg.D + ', n_heads=' + cfg.H + ', n_kv_heads=' + cfg.Hkv + (explicitDh ? ', head_dim=' + cfg.Dh : '') + '):');
  push('        super().__init__()');
  push(explicitDh ? '        assert n_heads % n_kv_heads == 0  # head_dim is set explicitly (head_dim * n_heads != d_model is fine)' : '        assert d_model % n_heads == 0 and n_heads % n_kv_heads == 0');
  push('        self.D, self.H, self.Hkv = d_model, n_heads, n_kv_heads');
  push(explicitDh ? '        self.Dh = head_dim' : '        self.Dh = d_model // n_heads');
  init.forEach((l) => push('        ' + l.text, l.blockId));
  push('');
  push('    def forward(self, x):');
  push('        B, T, D = x.shape');
  push('        H, Hkv, Dh = self.H, self.Hkv, self.Dh');
  fwd.forEach((l) => push('        ' + l.text, l.blockId));
  return { lines, text: lines.map((l) => l.text).join('\n') + '\n' };
}
