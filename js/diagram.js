// Academic-style architecture figures (Times, thin black outlines, pastel fills, data flows upward),
// generated as plain SVG so they can be exported as SVG or PNG.
// Figure text is always English, as it is typically used in papers.

const FONT = "'Times New Roman', Times, 'Liberation Serif', 'Nimbus Roman', serif";
const INK = '#1a1a1a';
export const FILL = {
  linear: '#F9D9B5', matmul: '#BFE0C8', scale: '#F8EDB2', mask: '#F4C9D2', softmax: '#C6DCF3',
  norm: '#F8EDB2', rope: '#D9D2F0', dropout: '#E4E4E4', concat: '#E4E4E4', custom: '#FFFFFF',
  attn: '#F9D9B5', ffn: '#C6DCF3', embed: '#D5E8C4', head: '#E5D3EE', moe: '#F4C9C0', cross: '#F9D9B5'
};

const esc = (s) => String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
const num = (n) => Math.round(n * 10) / 10;

function text(x, y, s, { size = 15, italic = false, anchor = 'middle', fill = INK, bold = false, halo = false } = {}) {
  const halos = halo ? ' stroke="#fff" stroke-width="3.5" paint-order="stroke" stroke-linejoin="round"' : '';
  return '<text x="' + num(x) + '" y="' + num(y) + '" font-family="' + FONT + '" font-size="' + size + '" text-anchor="' + anchor + '" fill="' + fill + '"' +
    (italic ? ' font-style="italic"' : '') + (bold ? ' font-weight="bold"' : '') + halos + '>' + esc(s) + '</text>';
}

const MARKER = '<defs><marker id="ah" markerWidth="9" markerHeight="9" refX="8" refY="4.5" orient="auto" markerUnits="userSpaceOnUse"><path d="M0,0.8 L8.5,4.5 L0,8.2 z" fill="' + INK + '"/></marker></defs>';

function svgWrap(w, h, body) {
  return '<svg xmlns="http://www.w3.org/2000/svg" width="' + num(w) + '" height="' + num(h) + '" viewBox="0 0 ' + num(w) + ' ' + num(h) + '">' +
    MARKER + '<rect width="100%" height="100%" fill="#ffffff"/>' + body + '</svg>';
}

function line(pts, { arrow = true, dashed = false } = {}) {
  return '<polyline points="' + pts.map((p) => num(p[0]) + ',' + num(p[1])).join(' ') + '" fill="none" stroke="' + INK + '" stroke-width="1.3" stroke-linejoin="round"' +
    (dashed ? ' stroke-dasharray="5 3"' : '') + (arrow ? ' marker-end="url(#ah)"' : '') + '/>';
}

function captionBlock(cx, y, lines) {
  return lines.map((l, i) => text(cx, y + i * 20, l, { size: i === 0 ? 15 : 13, italic: i > 0, fill: i === 0 ? INK : '#333' })).join('');
}

const estimateTextWidth = (s, size) => s.length * size * 0.47;

// ---------------------------------------------------------------------------
// 1. Attention-block figure built from the block program (Fig. 2 of "Attention Is All You Need" style)
// ---------------------------------------------------------------------------
const NORM_NAMES = { softmax: 'SoftMax', sparsemax: 'Sparsemax', relu2: 'ReLU² Norm.', sigmoid: 'Sigmoid' };

export function attentionDiagram(program, analysis, opts = {}) {
  const showShapes = opts.showShapes !== false;
  const nodes = [];
  const edges = [];
  const prod = new Map();
  const add = (label, sub, fill, kind = 'op', dashed = false) => {
    const n = { id: nodes.length, label, sub, fill, kind, dashed, parents: [] };
    nodes.push(n); return n;
  };
  const link = (name, to) => {
    const pr = prod.get(name);
    if (!pr) return;
    edges.push({ from: pr.node.id, to: to.id, name, shape: pr.shape });
    to.parents.push(pr.node.id);
  };
  const setProd = (name, node, blockId) => {
    const info = analysis.perBlock[blockId];
    const shape = info && info.out ? '[' + info.out.shape.join(', ') + ']' : '';
    prod.set(name, { node, shape });
  };
  const up = (s) => String(s).toUpperCase();

  program.blocks.forEach((b) => {
    const p = b.params;
    switch (b.type) {
      case 'input': { const n = add(p.name, null, '#fff', 'term'); setProd(p.name, n, b.id); break; }
      case 'proj': {
        const n = add(p.kind === 'lowrank' ? 'Low-rank Linear' : 'Linear', 'W_' + p.out + (p.kind === 'lowrank' ? ', r=' + p.rank : ''), FILL.linear);
        link(p.src, n); setProd(p.out, n, b.id); break;
      }
      case 'mla': {
        const dq = p.nope + p.rope;
        const nq = add(p.qRank > 0 ? 'Low-rank Linear' : 'Linear', 'W_' + (p.qRank > 0 ? 'DQ,UQ, r_q=' + p.qRank : 'Q'), FILL.linear);
        const nkv = add('Latent KV Linear', 'W_DKV,UKV, r_kv=' + p.kvRank, FILL.linear);
        const nr = add('Decoupled RoPE', 'd_r=' + p.rope + ', θ=' + p.base, FILL.rope);
        const nc = add('Concat', 'nope ‖ rope, d_h=' + dq, FILL.concat);
        link(p.src, nq); link(p.src, nkv); link(p.src, nr);
        [nq, nkv, nr].forEach((n) => {
          edges.push({ from: n.id, to: nc.id, name: n === nq ? 'q' : n === nkv ? 'c_kv' : 'rope', shape: '' }); nc.parents.push(n.id);
        });
        setProd(p.outQ, nc, b.id); setProd(p.outK, nc, b.id); setProd(p.outV, nkv, b.id);
        const info = analysis.perBlock[b.id];
        if (info && info.outs && info.outs.length === 3) {
          [p.outQ, p.outK, p.outV].forEach((nm, i) => { prod.get(nm).shape = '[' + info.outs[i].shape.join(', ') + ']'; });
        }
        break;
      }
      case 'rope': { const n = add('RoPE', 'θ=' + p.base, FILL.rope); link(p.target, n); setProd(p.target, n, b.id); break; }
      case 'score': {
        const n1 = add(p.kind === 'cosine' ? 'Cosine Sim.' : 'MatMul', up(p.q) + up(p.k) + 'ᵀ', FILL.matmul);
        link(p.q, n1); link(p.k, n1);
        let last = n1;
        if (p.scale === 'sqrt') {
          const qs = analysis.perBlock[b.id] && analysis.perBlock[b.id].ins[0] ? analysis.perBlock[b.id].ins[0].shape[3] : 'Dh';
          last = add('Scale', typeof qs === 'number' ? '1/√' + qs : '1/√d_h', FILL.scale);
          edges.push({ from: n1.id, to: last.id, name: p.out, shape: analysis.perBlock[b.id].out ? '[' + analysis.perBlock[b.id].out.shape.join(', ') + ']' : '' });
          last.parents.push(n1.id);
        }
        setProd(p.out, last, b.id); break;
      }
      case 'mask': {
        const sub = p.kind === 'causal' ? 'causal' : p.kind === 'sliding' ? 'window = ' + p.window : 'top-' + p.k;
        const n = add('Mask', sub, FILL.mask); link(p.target, n); setProd(p.target, n, b.id); break;
      }
      case 'norm': {
        const n = add(NORM_NAMES[p.fn] || p.fn, Number(p.temp) !== 1 ? 'τ = ' + p.temp : null, FILL.softmax);
        link(p.target, n); setProd(p.out, n, b.id); break;
      }
      case 'dropout': { const n = add('Dropout', 'p = ' + p.p, FILL.dropout); link(p.target, n); setProd(p.target, n, b.id); break; }
      case 'mix': {
        const n = add('MatMul', up(p.a) + up(p.v), FILL.matmul); link(p.a, n); link(p.v, n); setProd(p.out, n, b.id); break;
      }
      case 'outproj': {
        const c = add('Concat', 'heads', FILL.concat); link(p.src, c);
        const l = add('Linear', 'W_o', FILL.linear);
        edges.push({ from: c.id, to: l.id, name: 'concat', shape: '[B, T, D]' }); l.parents.push(c.id);
        setProd(p.out, l, b.id); break;
      }
      case 'ret': { const n = add('Output', null, '#fff', 'term'); link(p.src, n); break; }
      case 'custom': {
        const n = add(p.title, 'custom', FILL.custom, 'op', true);
        String(p.inputs || '').split(',').map((s) => s.trim()).filter(Boolean).forEach((nm) => link(nm, n));
        if (p.outName) setProd(p.outName, n, b.id);
        break;
      }
      default: break;
    }
  });

  if (!nodes.length) {
    return { svg: svgWrap(480, 160, text(240, 85, 'Add blocks to see the diagram', { size: 16, italic: true, fill: '#666' })), width: 480, height: 160 };
  }

  // Layering
  const W = 150, HOP = 48, HTERM = 34, HGAP = 54, VGAP = 84;
  const layer = nodes.map(() => 0);
  nodes.forEach((n) => { layer[n.id] = n.parents.length ? 1 + Math.max(...n.parents.map((q) => layer[q])) : 0; });
  const maxLayer = Math.max(...layer);
  const rows = [];
  for (let l = 0; l <= maxLayer; l++) rows.push(nodes.filter((n) => layer[n.id] === l));
  const cxOf = new Map();
  rows.forEach((row, l) => {
    const desired = row.map((n, i) => (n.parents.length ? n.parents.reduce((s, q) => s + cxOf.get(q), 0) / n.parents.length : (i - (row.length - 1) / 2) * (W + HGAP)));
    const order = row.map((n, i) => ({ n, d: desired[i] })).sort((a, b) => a.d - b.d);
    const xs = [];
    order.forEach((o, i) => { xs.push(i === 0 ? o.d : Math.max(o.d, xs[i - 1] + W + HGAP)); });
    const shift = order.reduce((s, o) => s + o.d, 0) / order.length - xs.reduce((s, x) => s + x, 0) / xs.length;
    order.forEach((o, i) => cxOf.set(o.n.id, xs[i] + shift));
  });
  const xsAll = nodes.map((n) => cxOf.get(n.id));
  const longEdges = edges.filter((e) => layer[e.to] - layer[e.from] > 1);
  const LABEL_W = showShapes ? 130 : 0;
  const minX = Math.min(...xsAll) - W / 2;
  const laneStart = Math.max(...xsAll) + W / 2 + 30;
  const maxX = Math.max(...xsAll) + W / 2 + (longEdges.length ? 30 + longEdges.length * 18 + LABEL_W : 0);
  const margin = 44;
  const capLines = [
    'Figure. ' + (opts.title || program.name || 'Attention block'),
    'B=' + program.config.B + ', T=' + program.config.T + ', d_model=' + program.config.D + ', h=' + program.config.H + (program.config.Hkv !== program.config.H ? ', h_kv=' + program.config.Hkv : '') + (program.config.Dh > 0 ? ', d_h=' + program.config.Dh : '')
  ];
  if (opts.subtitle) capLines.push(opts.subtitle);
  const capW = Math.max(...capLines.map((l, i) => estimateTextWidth(l, i ? 13 : 15))) + 40;
  const bodyW = maxX - minX;
  const width = Math.max(bodyW + margin * 2, capW, 420);
  const offX = (width - bodyW) / 2 - minX;
  const bodyH = maxLayer * VGAP + HOP;
  const height = bodyH + margin * 2 + 14 + capLines.length * 20;
  const baseCy = margin + bodyH - HOP / 2;
  const cyOf = (n) => baseCy - layer[n.id] * VGAP;
  const hOf = (n) => (n.kind === 'term' ? HTERM : HOP);
  const X = (n) => cxOf.get(n.id) + offX;

  // Distribute ports so that several edges entering/leaving one node do not share a point.
  const PORT = 34;
  const outs = new Map(), ins = new Map();
  edges.forEach((e) => {
    if (!outs.has(e.from)) outs.set(e.from, []);
    if (!ins.has(e.to)) ins.set(e.to, []);
    outs.get(e.from).push(e); ins.get(e.to).push(e);
  });
  outs.forEach((list, id) => { list.sort((a, b) => cxOf.get(a.to) - cxOf.get(b.to)); list.forEach((e, i) => { e.sOff = (i - (list.length - 1) / 2) * PORT; }); });
  ins.forEach((list, id) => { list.sort((a, b) => cxOf.get(a.from) - cxOf.get(b.from)); list.forEach((e, i) => { e.tOff = (i - (list.length - 1) / 2) * PORT; }); });

  const labelBox = (x, y, s) => {
    const w = s.length * 5.2 + 8;
    return '<rect x="' + num(x - w / 2) + '" y="' + num(y - 10) + '" width="' + num(w) + '" height="15" fill="#fff"/>' + text(x, y + 1, s, { size: 11, italic: true });
  };

  let out = '';
  edges.forEach((e) => {
    const s = nodes[e.from], t = nodes[e.to];
    const sx = X(s) + e.sOff, sy = cyOf(s) - hOf(s) / 2;
    const tx = X(t) + e.tOff, ty = cyOf(t) + hOf(t) / 2;
    const diff = layer[e.to] - layer[e.from];
    const label = e.name + ' ' + e.shape;
    if (diff > 1) {
      const idx = longEdges.indexOf(e);
      const lx = laneStart + offX + idx * 18;
      const y1 = sy - 14 - idx * 5, y2 = ty + 16 + idx * 5;
      out += line([[sx, sy], [sx, y1], [lx, y1], [lx, y2], [tx, y2], [tx, ty]]);
      if (showShapes) out += text(lx + 6, (y1 + y2) / 2, label, { size: 11, italic: true, anchor: 'start' });
    } else {
      const d = (sy - ty) / 2;
      out += '<path d="M' + num(sx) + ',' + num(sy) + ' C' + num(sx) + ',' + num(sy - d) + ' ' + num(tx) + ',' + num(ty + d) + ' ' + num(tx) + ',' + num(ty) + '" fill="none" stroke="' + INK + '" stroke-width="1.3" marker-end="url(#ah)"/>';
      if (showShapes) {
        if (Math.abs(sx - tx) < 8) {
          // Several edges between the same pair of nodes: put their labels on opposite sides so they do not collide.
          const shared = ins.get(e.to).length > 1 || outs.get(e.from).length > 1;
          const left = shared && (e.tOff || 0) < 0;
          out += text((sx + tx) / 2 + (left ? -7 : 7), (sy + ty) / 2 + 4, label, { size: 11, italic: true, anchor: left ? 'end' : 'start' });
        }
        else out += labelBox((sx + tx) / 2, (sy + ty) / 2 + 3, label);
      }
    }
  });
  nodes.forEach((n) => {
    const cx = X(n), cy = cyOf(n), h = hOf(n);
    out += '<rect x="' + num(cx - W / 2) + '" y="' + num(cy - h / 2) + '" width="' + W + '" height="' + h + '" rx="' + (n.kind === 'term' ? h / 2 : 8) + '" fill="' + n.fill + '" stroke="' + INK + '" stroke-width="1.3"' + (n.dashed ? ' stroke-dasharray="5 3"' : '') + '/>';
    if (n.sub) { out += text(cx, cy - 3, n.label, { size: 15 }) + text(cx, cy + 13, n.sub, { size: 12, italic: true, fill: '#333' }); }
    else out += text(cx, cy + 5, n.label, { size: n.kind === 'term' ? 15 : 16, italic: n.kind === 'term' });
  });
  out += captionBlock(width / 2, height - 16 - (capLines.length - 1) * 20, capLines);
  return { svg: svgWrap(width, height, out), width, height };
}

// ---------------------------------------------------------------------------
// 2. Whole-model figure from a Hugging Face config spec (Fig. 1 style)
// ---------------------------------------------------------------------------
const BW = 250, GAP = 30;

function elHeight(e) {
  if (e.t === 'term') return 34;
  if (e.t === 'add') return 24;
  return 36 + 17 * (e.subs ? e.subs.length : 0);
}

function buildColumn(seq, cx) {
  let u = 0;
  const placed = [], frames = [], skips = [], anchors = {}, fs = [], ss = [];
  seq.forEach((e) => {
    if (e.t === 'frameStart') { fs.push({ u0: u - GAP + 8, label: e.text }); return; }
    if (e.t === 'frameEnd') { const f = fs.pop(); frames.push({ u0: f.u0, u1: u - 8, label: f.label }); return; }
    if (e.t === 'skipStart') { ss.push(u - GAP / 2); return; }
    if (e.t === 'skipEnd') {
      const u0 = ss.pop(), last = placed[placed.length - 1];
      skips.push({ u0, u1: last.cu, endX: last.t === 'add' ? cx + 12 : cx + BW / 2 });
      return;
    }
    if (e.t === 'anchor') { anchors[e.id] = placed[placed.length - 1]; return; }
    const h = elHeight(e);
    placed.push(Object.assign({}, e, { u0: u, h, cu: u + h / 2 }));
    u += h + GAP;
  });
  return { placed, frames, skips, anchors, cx, topU: u - GAP };
}

function renderColumn(col, Y) {
  const { cx } = col;
  let out = '';
  col.frames.forEach((f) => {
    out += '<rect x="' + num(cx - BW / 2 - 20) + '" y="' + num(Y(f.u1)) + '" width="' + (BW + 66) + '" height="' + num(f.u1 - f.u0) + '" rx="12" fill="none" stroke="#555" stroke-width="1.2"/>';
    out += text(cx - BW / 2 - 28, Y(f.u1) + 24, f.label, { size: 20, anchor: 'end' });
  });
  col.skips.forEach((s) => {
    const xr = cx + BW / 2 + 24;
    out += line([[cx, Y(s.u0)], [xr, Y(s.u0)], [xr, Y(s.u1)], [s.endX, Y(s.u1)]]);
  });
  col.placed.forEach((e, i) => {
    if (i > 0) { const p = col.placed[i - 1]; out += line([[cx, Y(p.u0 + p.h)], [cx, Y(e.u0)]]); }
    const top = Y(e.u0 + e.h);
    if (e.t === 'add') {
      const cy = Y(e.cu);
      out += '<circle cx="' + num(cx) + '" cy="' + num(cy) + '" r="12" fill="#fff" stroke="' + INK + '" stroke-width="1.3"/>' +
        '<path d="M' + (cx - 6) + ',' + num(cy) + ' H' + (cx + 6) + ' M' + cx + ',' + num(cy - 6) + ' V' + num(cy + 6) + '" stroke="' + INK + '" stroke-width="1.3"/>';
    } else {
      out += '<rect x="' + (cx - BW / 2) + '" y="' + num(top) + '" width="' + BW + '" height="' + e.h + '" rx="' + (e.t === 'term' ? 17 : 8) + '" fill="' + (e.t === 'term' ? '#fff' : e.fill) + '" stroke="' + INK + '" stroke-width="1.3"/>';
      if (e.t === 'term') out += text(cx, top + 22, e.text, { size: 15, italic: true });
      else {
        out += text(cx, top + (e.subs && e.subs.length ? 22 : 23), e.text, { size: 16 });
        (e.subs || []).forEach((s, k) => { out += text(cx, top + 39 + 15 * k, s, { size: 12, italic: true, fill: '#333' }); });
      }
    }
  });
  return out;
}

const act = (a) => {
  const s = String(a).toLowerCase();
  if (/silu|swish/.test(s)) return 'SwiGLU';
  if (/gelu/.test(s)) return 'GeGLU';
  if (/relu/.test(s)) return 'ReLU';
  return a;
};
const mlpAct = (a) => {
  const s = String(a).toLowerCase();
  if (/gelu/.test(s)) return 'GELU';
  if (/relu/.test(s)) return 'ReLU';
  if (/silu|swish/.test(s)) return 'SiLU';
  return a;
};
const UNKNOWN_EN = { pos: 'positional scheme', norm: 'normalisation type', topk: 'experts per token', ffn: 'FFN width', mla_dims: 'MLA head dims' };
const fmtN = (n) => (n == null ? '—' : Number(n).toLocaleString('en-US'));
const normName = (s) => (s.norm === 'rms' ? 'RMSNorm' : s.norm === 'layer' ? 'LayerNorm' : 'Norm (unknown type)');

// "4, 8, 12, …, 92, 93" style summary of the 1-based indices of layers of one kind.
function layerList(kinds, kind) {
  const idx = kinds.map((k, i) => (k === kind ? i + 1 : null)).filter((x) => x !== null);
  if (idx.length <= 7) return idx.join(', ');
  return idx.slice(0, 3).join(', ') + ', …, ' + idx.slice(-2).join(', ');
}

function mlaSubs(s) {
  const m = s.mla;
  const subs = ['r_q=' + (m.qLoraRank || '—') + ', r_kv=' + m.kvLoraRank + ', h=' + s.heads];
  subs.push(m.qkNope != null && m.qkRope != null ? 'd_nope=' + m.qkNope + ', d_rope=' + m.qkRope + ', d_v=' + (m.vHead == null ? '?' : m.vHead) : 'head dims: unknown');
  return subs;
}

function hybridAttnBox(s) {
  const hy = s.hybrid;
  const subs = [];
  const fullName = s.attnKind === 'MLA' ? 'MLA' : s.attnKind === 'MHA' ? 'MHA' : s.attnKind;
  if (hy.linear) subs.push(hy.linear + ' × ' + hy.linearName + ' (linear attention)');
  if (hy.sliding) subs.push(hy.sliding + ' × sliding-window' + (hy.window ? ' (' + fmtN(hy.window) + ')' : '') + ' attention');
  if (hy.full) subs.push(hy.full + ' × full attention (' + fullName + ')');
  if (hy.other) subs.push(hy.other + ' × unspecified layer type');
  if (hy.full) subs.push('full-attn layers: ' + layerList(hy.kinds, 'full'));
  if (s.mla) subs.push(...mlaSubs(s)); else subs.push('h=' + s.heads + (s.kvHeads !== s.heads ? ', h_kv=' + s.kvHeads : '') + ', d_h=' + Math.round(s.headDim));
  const extra = posLabel(s);
  if (extra) subs.push(extra);
  return { t: 'box', text: 'Hybrid Attention (per-layer type)', subs, fill: FILL.attn };
}

function posLabel(s) {
  if (s.pos === 'rope') return 'RoPE' + (s.ropeTheta ? ' θ=' + fmtN(s.ropeTheta) : '') + (s.mla ? ' (decoupled)' : '');
  if (s.pos === 'alibi') return 'ALiBi';
  if (s.pos === 'relative') return 'relative pos. bias';
  if (s.pos === 'unknown') return 'positional scheme: unknown';
  return '';
}

function attnBox(s, { causal, cross = false }) {
  if (s.hybrid && !cross) return hybridAttnBox(s);
  const kind = s.attnKind;
  const kindName = kind === 'MHA' ? 'Multi-Head' : kind === 'MQA' ? 'Multi-Query' : kind === 'GQA' ? 'Grouped-Query' : 'Multi-head Latent';
  const title = (cross ? 'Cross-' : causal ? 'Masked ' : '') + (cross ? 'Attention' : kindName + ' Attention');
  const subs = [];
  if (kind === 'MLA') subs.push(...mlaSubs(s));
  else subs.push('h=' + s.heads + (kind === 'GQA' || kind === 'MQA' ? ', h_kv=' + s.kvHeads : '') + ', d_h=' + Math.round(s.headDim));
  if (!cross) {
    const extra = [];
    const pl = posLabel(s);
    if (pl && s.pos !== 'learned') extra.push(pl);
    if (s.window) extra.push('window=' + fmtN(s.window));
    if (extra.length) subs.push(extra.join(', '));
  }
  return { t: 'box', text: title, subs, fill: cross ? FILL.cross : FILL.attn };
}

const dff = (n) => (n == null ? 'unknown' : fmtN(n));
function ffnBox(s) {
  if (s.ffnType === 'moe') {
    const subs = ['E=' + s.experts + ', top-' + (s.topK || '?') + (s.shared ? ', shared=' + s.shared : ''), 'd_ff=' + dff(s.ffn) + ' / expert'];
    if (s.denseLayers > 0) subs.push('first ' + s.denseLayers + ' layer' + (s.denseLayers > 1 ? 's' : '') + ': dense FFN' + (s.denseFfn ? ' (d_ff=' + fmtN(s.denseFfn) + ')' : ''));
    return { t: 'box', text: 'Mixture of Experts', subs, fill: FILL.moe };
  }
  if (s.ffnType === 'gated') return { t: 'box', text: act(s.act) + ' Feed-Forward', subs: ['d_ff=' + dff(s.ffn)], fill: FILL.ffn };
  return { t: 'box', text: 'Feed-Forward', subs: [mlpAct(s.act) + ', d_ff=' + dff(s.ffn)], fill: FILL.ffn };
}

function sublayer(s, box, anchorId) {
  // Returns the element sequence for one residual sub-layer (norm placement aware).
  const seq = [{ t: 'skipStart' }];
  const anchor = anchorId ? [{ t: 'anchor', id: anchorId }] : [];
  if (s.placement === 'pre') {
    seq.push({ t: 'box', text: normName(s), fill: FILL.norm }, box, ...anchor, { t: 'add' }, { t: 'skipEnd' });
  } else {
    seq.push(box, ...anchor, { t: 'box', text: 'Add & Norm', fill: FILL.norm }, { t: 'skipEnd' });
  }
  return seq;
}

function headFor(s) {
  const a = s.arch || '';
  if (/ForMaskedLM|MaskedLM/.test(a)) return [{ t: 'box', text: 'MLM Head', subs: ['Linear + softmax'], fill: FILL.head }, { t: 'term', text: 'Token probabilities' }];
  if (/SequenceClassification/.test(a)) return [{ t: 'box', text: 'Pooler + Classifier', fill: FILL.head }, { t: 'term', text: 'Class logits' }];
  if (/TokenClassification/.test(a)) return [{ t: 'box', text: 'Token Classifier', fill: FILL.head }, { t: 'term', text: 'Tag logits' }];
  return [{ t: 'term', text: 'Hidden states' }];
}

function embedBox(s) {
  const learned = s.pos === 'learned';
  const subs = [(s.vocab ? 'V=' + fmtN(s.vocab) + ', ' : '') + 'd=' + fmtN(s.hidden)];
  if (s.pos === 'unknown') subs.push('positional scheme: unknown');
  return { t: 'box', text: learned ? 'Token + Position Embedding' : 'Token Embedding', subs, fill: FILL.embed };
}

function decoderSeq(s, { cross, layers, inputLabel, finalHead, causal }) {
  const seq = [{ t: 'term', text: inputLabel }, embedBox(s), { t: 'frameStart', text: layers + '×' }];
  seq.push(...sublayer(s, attnBox(s, { causal })));
  if (cross) seq.push(...sublayer(s, attnBox(s, { causal: false, cross: true }), 'cross'));
  seq.push(...sublayer(s, ffnBox(s)));
  seq.push({ t: 'frameEnd' });
  if (s.placement === 'pre') seq.push({ t: 'box', text: 'Final ' + normName(s), fill: FILL.norm });
  seq.push(...finalHead);
  return seq;
}

export function modelDiagram(spec) {
  const s = spec;
  const margin = 40;
  let cols = [];
  let crossLink = null;
  const famLabel = { decoder: 'decoder-only', encoder: 'encoder-only', encdec: 'encoder–decoder' }[s.family];
  const layersTxt = s.family === 'encdec' ? 'L_enc=' + s.encLayers + ', L_dec=' + s.decLayers : 'L=' + s.layers;
  const capLines = [
    'Figure. Architecture of ' + s.name + ' (' + famLabel + ' Transformer).',
    layersTxt + ', d_model=' + fmtN(s.hidden) + ', h=' + s.heads + (s.attnKind === 'GQA' || s.attnKind === 'MQA' ? ', h_kv=' + s.kvHeads : '') +
      (s.ffnType === 'moe' ? ', E=' + s.experts + (s.topK ? ', top-' + s.topK : '') : ', d_ff=' + dff(s.ffn)) + (s.vocab ? ', V=' + fmtN(s.vocab) : '') + (s.params ? ', ≈' + formatParamsShort(s.params) + ' params' : '')
  ];
  if (s.hybrid) {
    const hy = s.hybrid, parts = [];
    if (hy.linear) parts.push(hy.linear + ' ' + hy.linearName + ' (linear)');
    if (hy.sliding) parts.push(hy.sliding + ' sliding-window');
    if (hy.full) parts.push(hy.full + ' full-attention' + (s.mla ? ' (MLA)' : ''));
    capLines.push('Layers are not identical: ' + parts.join(' + ') + (s.denseLayers ? '; first ' + s.denseLayers + ' FFN layer(s) dense' : '') + '. The repeated block is drawn once.');
  } else if (s.denseLayers) capLines.push('First ' + s.denseLayers + ' layer(s) use a dense FFN; the remaining layers are MoE.');
  if (s.multimodal) capLines.push('Text backbone of a multimodal model (' + s.multimodal + ').');
  (s.notes || []).filter((n) => n.key !== 'hf.dimnote').forEach((n) => capLines.push('Note: ' + n.en));
  if ((s.unknown || []).length) capLines.push('Not read from config.json (shown as unknown): ' + s.unknown.map((u) => UNKNOWN_EN[u] || u).join(', ') + '.');
  const ncol = s.family === 'encdec' ? 2 : 1;
  const bodyW = ncol * BW + (ncol - 1) * 190 + 140;
  const capW = Math.max(...capLines.map((l, i) => estimateTextWidth(l, i ? 13 : 15))) + 40;
  const width = Math.max(bodyW + margin * 2, capW);
  const dx = (width - (bodyW + margin * 2)) / 2;
  const colX = (i) => dx + margin + 70 + BW / 2 + i * (BW + 190);

  if (s.family === 'encdec') {
    const encSeq = [{ t: 'term', text: 'Inputs' }, embedBox(s), { t: 'frameStart', text: s.encLayers + '×' }, ...sublayer(s, attnBox(s, { causal: false })), ...sublayer(s, ffnBox(s)), { t: 'frameEnd' }];
    if (s.placement === 'pre') encSeq.push({ t: 'box', text: 'Final ' + normName(s), fill: FILL.norm });
    const decHead = [{ t: 'box', text: 'Linear', subs: [s.tie === true ? 'tied with embedding' : 'LM head'], fill: FILL.head }, { t: 'box', text: 'Softmax', fill: FILL.softmax }, { t: 'term', text: 'Output probabilities' }];
    const decSeq = decoderSeq(s, { cross: true, layers: s.decLayers, inputLabel: 'Outputs (shifted right)', finalHead: decHead, causal: true });
    cols = [buildColumn(encSeq, colX(0)), buildColumn(decSeq, colX(1))];
    crossLink = true;
  } else if (s.family === 'encoder') {
    const seq = [{ t: 'term', text: 'Input tokens' }, embedBox(s), { t: 'frameStart', text: s.layers + '×' }, ...sublayer(s, attnBox(s, { causal: false })), ...sublayer(s, ffnBox(s)), { t: 'frameEnd' }];
    if (s.placement === 'pre') seq.push({ t: 'box', text: 'Final ' + normName(s), fill: FILL.norm });
    seq.push(...headFor(s));
    cols = [buildColumn(seq, colX(0))];
  } else {
    const head = [{ t: 'box', text: 'Linear', subs: [s.tie === true ? 'LM head, tied with embedding' : 'LM head'], fill: FILL.head }, { t: 'box', text: 'Softmax', fill: FILL.softmax }, { t: 'term', text: 'Output probabilities' }];
    cols = [buildColumn(decoderSeq(s, { cross: false, layers: s.layers, inputLabel: 'Input token ids', finalHead: head, causal: true }), colX(0))];
  }

  const bodyH = Math.max(...cols.map((c) => c.topU)) + 20;
  const height = margin + bodyH + 30 + capLines.length * 20 + 10;
  const yBase = margin + bodyH;
  const Y = (uu) => yBase - uu;
  let body = '';
  cols.forEach((c) => { body += renderColumn(c, Y); });
  if (crossLink) {
    const enc = cols[0], dec = cols[1];
    const encTop = enc.placed[enc.placed.length - 1];
    const cr = dec.anchors.cross;
    const yTop = Y(encTop.u0 + encTop.h);
    const midX = (enc.cx + BW / 2 + 46 + dec.cx - BW / 2 - 24) / 2;
    body += line([[enc.cx, yTop], [enc.cx, yTop - 20], [midX, yTop - 20], [midX, Y(cr.cu)], [dec.cx - BW / 2, Y(cr.cu)]]);
  }
  body += captionBlock(width / 2, yBase + 42, capLines);
  return { svg: svgWrap(width, height, body), width, height };
}

function formatParamsShort(n) {
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + 'M';
  return (n / 1e3).toFixed(0) + 'K';
}

// ---------------------------------------------------------------------------
// PNG export (browser only)
// ---------------------------------------------------------------------------
export function svgToPngBlob(svg, width, height, scale = 3) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const url = URL.createObjectURL(new Blob([svg], { type: 'image/svg+xml;charset=utf-8' }));
    img.onload = () => {
      const canvas = document.createElement('canvas');
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const ctx = canvas.getContext('2d');
      ctx.fillStyle = '#ffffff';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
      URL.revokeObjectURL(url);
      canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('png'))), 'image/png');
    };
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('svg')); };
    img.src = url;
  });
}
