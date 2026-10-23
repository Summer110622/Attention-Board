import { BLOCKS, CATEGORIES, PALETTE, DEFAULT_CONFIG, analyze, generateCode, newBlock, fmtShape } from './blocks.js';
import { TEMPLATES } from './templates.js';
import { evaluate, attentionStats, isAttentionMap } from './tensor.js';
import { fetchHfConfig, configToSpec, specToProgram, formatParams } from './hf.js';
import { attentionDiagram, modelDiagram, svgToPngBlob } from './diagram.js';
import { sanitizeProgram, encodeShare, decodeShare } from './storage.js';
import { t, setLang, getLang, detectLang } from './i18n.js';
import { h, icon, logo, downloadBlob, toast, safeName } from './ui.js';

const STORAGE_KEY = 'attention-board:program:v1';

const state = {
  program: null, selId: null, tab: 'props', paletteTab: 'blocks', search: '', view: 'blocks',
  showShapes: true, pngScale: 3, past: [], future: [], previewBlock: null, previewHead: 0, seed: 7,
  hf: { url: '', busy: false, error: null, spec: null, pasted: '', note: null }
};
let analysis = null;
let dragInfo = null;

// ---------------------------------------------------------------------------
// State management
// ---------------------------------------------------------------------------
// DOM append() would render null as the text "null"; filter empty children first.
function add(parent, ...kids) { parent.append(...kids.flat(Infinity).filter((k) => k !== null && k !== undefined && k !== false)); }

function save() {
  try { localStorage.setItem(STORAGE_KEY, JSON.stringify(state.program)); } catch (e) { /* storage may be unavailable */ }
}

function recompute() {
  analysis = analyze(state.program);
}

function commit(mutator) {
  state.past.push(JSON.stringify(state.program));
  if (state.past.length > 100) state.past.shift();
  state.future = [];
  mutator(state.program);
  recompute();
  save();
  renderAll();
}

function undo() {
  if (!state.past.length) return;
  state.future.push(JSON.stringify(state.program));
  state.program = JSON.parse(state.past.pop());
  recompute(); save(); renderAll();
}
function redo() {
  if (!state.future.length) return;
  state.past.push(JSON.stringify(state.program));
  state.program = JSON.parse(state.future.pop());
  recompute(); save(); renderAll();
}

function replaceProgram(program) {
  commit((p) => { Object.assign(p, program); });
  state.selId = state.program.blocks[0] ? state.program.blocks[0].id : null;
  renderAll();
}

function uniqueName(base, scope) {
  if (!scope.includes(base)) return base;
  let i = 2;
  while (scope.includes(base + i)) i++;
  return base + i;
}

function scopeAt(index) {
  const blocks = state.program.blocks;
  if (index < blocks.length) return analysis.perBlock[blocks[index].id].scope;
  return Array.from(new Set(analysis.tensors.map((x) => x.name)));
}

function smartBlock(type, index) {
  const def = BLOCKS[type];
  const scope = scopeAt(index);
  const params = def.defaults();
  def.fields.forEach((f) => {
    if (f.kind === 'tensor' && !scope.includes(params[f.key]) && scope.length) params[f.key] = scope[scope.length - 1];
  });
  ['out', 'outName'].forEach((k) => { if (typeof params[k] === 'string' && type !== 'custom') params[k] = uniqueName(params[k], scope); });
  return newBlock(type, params);
}

function insertBlock(type, index) {
  const b = smartBlock(type, index);
  commit((p) => { p.blocks.splice(index, 0, b); });
  state.selId = b.id;
  renderAll();
}

function moveBlock(id, toIndex) {
  commit((p) => {
    const from = p.blocks.findIndex((b) => b.id === id);
    if (from < 0) return;
    const [b] = p.blocks.splice(from, 1);
    p.blocks.splice(toIndex > from ? toIndex - 1 : toIndex, 0, b);
  });
}

function removeBlock(id) {
  commit((p) => { p.blocks = p.blocks.filter((b) => b.id !== id); });
  if (state.selId === id) state.selId = null;
  renderAll();
}

function selIndex() { return state.program.blocks.findIndex((b) => b.id === state.selId); }
function selBlock() { return state.program.blocks.find((b) => b.id === state.selId) || null; }

// ---------------------------------------------------------------------------
// Containers
// ---------------------------------------------------------------------------
let els = {};

function mount() {
  const app = document.getElementById('app');
  app.replaceChildren();
  els = {
    top: h('header', { class: 'topbar' }),
    paletteHead: h('div', { class: 'palette-head' }),
    paletteBody: h('div', { class: 'palette-body' }),
    paletteFoot: h('div', { class: 'palette-foot' }),
    toolbar: h('div', { class: 'toolbar' }),
    canvas: h('div', { class: 'canvas' }),
    inspTabs: h('div', { class: 'tabs', role: 'tablist' }),
    inspBody: h('div', { class: 'insp-body' }),
    status: h('footer', { class: 'status' })
  };
  app.append(
    els.top,
    h('div', { class: 'main' },
      h('aside', { class: 'palette' }, els.paletteHead, els.paletteBody, els.paletteFoot),
      h('main', { class: 'center' }, els.toolbar, els.canvas),
      h('aside', { class: 'inspector' }, els.inspTabs, els.inspBody)),
    els.status
  );
  renderPaletteHead();
  renderAll();
}

function renderAll() {
  renderTop(); renderPalette(); renderToolbar(); renderCanvas(); renderInspector(); renderStatus();
}

// ---------------------------------------------------------------------------
// Top bar
// ---------------------------------------------------------------------------
function renderTop() {
  const nameInput = h('input', {
    class: 'name-input', value: state.program.name, 'aria-label': t('project.name'),
    onchange: (e) => commit((p) => { p.name = e.target.value.trim() || 'Untitled attention'; })
  });
  els.top.replaceChildren(
    h('div', { class: 'brand' }, logo(), t('app.title')),
    h('span', { class: 'sep' }), nameInput,
    h('span', { class: 'saved' }, t('saved')),
    h('span', { class: 'spacer' }),
    h('button', { class: 'btn icon', 'aria-label': t('btn.undo'), title: t('btn.undo'), disabled: !state.past.length, onclick: undo }, icon('undo', 18)),
    h('button', { class: 'btn icon', 'aria-label': t('btn.redo'), title: t('btn.redo'), disabled: !state.future.length, onclick: redo }, icon('redo', 18)),
    h('button', { class: 'btn', onclick: shareLink }, icon('link'), t('btn.share')),
    h('button', { class: 'btn', onclick: exportJson }, icon('download'), t('btn.export')),
    h('button', { class: 'btn', onclick: () => document.getElementById('file-input').click() }, icon('upload'), t('btn.import')),
    h('button', { class: 'btn', onclick: () => { setLang(getLang() === 'ja' ? 'en' : 'ja'); mount(); } }, t('lang.switch')),
    h('button', { class: 'btn primary', onclick: openHf }, icon('hf'), t('btn.hf'))
  );
}

async function shareLink() {
  const url = location.origin + location.pathname + '#p=' + encodeShare(state.program);
  try { await navigator.clipboard.writeText(url); toast(t('toast.shared')); } catch (e) { window.prompt('URL', url); }
}
function exportJson() {
  downloadBlob(new Blob([JSON.stringify(state.program, null, 2)], { type: 'application/json' }), safeName(state.program.name) + '.attention.json');
}
function importJsonFile(file) {
  const reader = new FileReader();
  reader.onload = () => {
    try {
      const prog = sanitizeProgram(JSON.parse(reader.result));
      replaceProgram(prog);
      toast(t('toast.imported'));
    } catch (e) { toast(t('toast.badfile')); }
  };
  reader.readAsText(file);
}

// ---------------------------------------------------------------------------
// Palette
// ---------------------------------------------------------------------------
function renderPaletteHead() {
  const seg = h('div', { class: 'segmented', role: 'tablist' },
    h('button', { id: 'pt-blocks', role: 'tab', 'aria-selected': state.paletteTab === 'blocks', onclick: () => { state.paletteTab = 'blocks'; syncPaletteHead(); renderPalette(); } }, t('tab.blocks')),
    h('button', { id: 'pt-templates', role: 'tab', 'aria-selected': state.paletteTab === 'templates', onclick: () => { state.paletteTab = 'templates'; syncPaletteHead(); renderPalette(); } }, t('tab.templates')));
  const search = h('label', { class: 'search' }, icon('search', 16, '#4A5568'),
    h('span', { class: 'sr-only' }, t('search.placeholder')),
    h('input', { id: 'palette-search', placeholder: t('search.placeholder'), value: state.search, oninput: (e) => { state.search = e.target.value; renderPalette(); } }));
  els.paletteHead.replaceChildren(seg, search);
}
function syncPaletteHead() {
  document.getElementById('pt-blocks').setAttribute('aria-selected', state.paletteTab === 'blocks');
  document.getElementById('pt-templates').setAttribute('aria-selected', state.paletteTab === 'templates');
  document.getElementById('palette-search').parentElement.style.display = state.paletteTab === 'blocks' ? '' : 'none';
}

function renderPalette() {
  syncPaletteHead();
  const body = els.paletteBody;
  const scroll = body.scrollTop;
  if (state.paletteTab === 'templates') {
    body.replaceChildren(
      h('p', { class: 'hint' }, t('tpl.warn')),
      ...TEMPLATES.map((tp) => h('button', { class: 'tpl', onclick: () => { const prog = tp.make(); replaceProgram(prog); } }, t('tpl.' + tp.id)))
    );
    els.paletteFoot.replaceChildren();
    body.scrollTop = scroll;
    return;
  }
  const q = state.search.trim().toLowerCase();
  const groups = PALETTE.map((g) => {
    const cat = CATEGORIES[g.cat];
    const chips = g.types.filter((ty) => !q || t('block.' + ty + '.name').toLowerCase().includes(q) || ty.includes(q))
      .map((ty) => {
        const el = h('button', {
          class: 'chip', draggable: 'true', title: t('block.' + ty + '.desc'),
          onclick: () => { const i = selIndex(); insertBlock(ty, i >= 0 ? i + 1 : state.program.blocks.length); },
          ondragstart: (e) => { dragInfo = { from: 'palette', type: ty }; e.dataTransfer.setData('text/plain', 'ab:' + ty); e.dataTransfer.effectAllowed = 'copy'; },
          ondragend: () => { dragInfo = null; clearDropzones(); }
        }, t('block.' + ty + '.name'));
        return el;
      });
    if (!chips.length) return null;
    const c = h('section', { class: 'cat', style: '--c:' + cat.color + ';--t:' + cat.tint },
      h('h3', {}, h('span', { class: 'dot' }), t('cat.' + g.cat)),
      h('div', { class: 'chips' }, chips));
    return c;
  });
  body.replaceChildren(h('p', { class: 'hint' }, t('palette.hint')), ...groups);
  els.paletteFoot.replaceChildren(
    h('button', { class: 'dashed', onclick: () => { const i = selIndex(); insertBlock('custom', i >= 0 ? i + 1 : state.program.blocks.length); } }, icon('plus'), t('palette.custom')));
  body.scrollTop = scroll;
}

// ---------------------------------------------------------------------------
// Center: toolbar + script / diagram
// ---------------------------------------------------------------------------
function renderToolbar() {
  const seg = h('div', { class: 'segmented' },
    h('button', { 'aria-pressed': state.view === 'blocks', onclick: () => { state.view = 'blocks'; renderToolbar(); renderCanvas(); } }, t('view.blocks')),
    h('button', { 'aria-pressed': state.view === 'diagram', onclick: () => { state.view = 'diagram'; renderToolbar(); renderCanvas(); } }, t('view.diagram')));
  const kids = [seg, h('span', { class: 'spacer' })];
  if (state.view === 'diagram') {
    kids.push(
      h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: state.showShapes, onchange: (e) => { state.showShapes = e.target.checked; renderCanvas(); } }), t('fig.shapes')),
      h('label', { class: 'check' }, t('fig.scale'),
        h('select', { 'aria-label': t('fig.scale'), onchange: (e) => { state.pngScale = Number(e.target.value); } },
          [2, 3, 4].map((n) => h('option', { value: n, selected: n === state.pngScale }, n + '×')))),
      h('button', { class: 'btn small', onclick: () => exportDiagram('svg') }, icon('download', 14), t('btn.svg')),
      h('button', { class: 'btn small primary', onclick: () => exportDiagram('png') }, icon('image', 14), t('btn.png'))
    );
  } else {
    kids.push(h('button', { class: 'btn small', onclick: () => { if (state.program.blocks.length) commit((p) => { p.blocks = []; }); state.selId = null; renderAll(); } }, icon('trash', 14), t('btn.clear')));
  }
  els.toolbar.replaceChildren(...kids);
}

function clearDropzones() {
  document.querySelectorAll('.dz.over').forEach((d) => d.classList.remove('over'));
}

function handleDrop(index, e) {
  e.preventDefault();
  clearDropzones();
  const info = dragInfo;
  dragInfo = null;
  if (!info) return;
  if (info.from === 'palette') insertBlock(info.type, index);
  else if (info.from === 'script') { moveBlock(info.id, index); state.selId = info.id; renderAll(); }
}

function dropzone(index) {
  return h('div', {
    class: 'dz', 'data-label': t('drop.here'),
    ondragover: (e) => { if (!dragInfo) return; e.preventDefault(); e.dataTransfer.dropEffect = dragInfo.from === 'palette' ? 'copy' : 'move'; e.currentTarget.classList.add('over'); },
    ondragleave: (e) => e.currentTarget.classList.remove('over'),
    ondrop: (e) => handleDrop(index, e)
  });
}

function blockEl(b, i) {
  const def = BLOCKS[b.type];
  const cat = CATEGORIES[def.cat];
  const info = analysis.perBlock[b.id];
  const sel = state.selId === b.id;
  const bad = info.errors.length > 0;
  const parts = def.parts(b.params).map(([txt, pill]) => h('span', { class: pill ? 'pill' : 'w' }, txt));
  const tool = (ic, label, fn, disabled) => h('button', { class: 'tool', 'aria-label': label, title: label, disabled, onclick: (e) => { e.stopPropagation(); fn(); } }, icon(ic, 14));
  const blocks = state.program.blocks;
  const el = h('div', {
    class: 'block' + (sel ? ' selected' : '') + (bad ? ' invalid' : '') + (b.type === 'input' && i === 0 ? ' first-hat' : ''),
    style: '--c:' + cat.color + ';--t:' + cat.tint,
    tabindex: '0', role: 'button', 'aria-pressed': sel, draggable: 'true', 'data-id': b.id,
    'aria-label': t('block.' + b.type + '.name') + ': ' + def.parts(b.params).map((p) => p[0]).join(' '),
    onclick: () => { state.selId = b.id; renderCanvas(); renderInspector(); },
    onkeydown: (e) => {
      if (e.target !== e.currentTarget) return;
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); state.selId = b.id; renderCanvas(); renderInspector(); }
      else if (e.key === 'Delete' || e.key === 'Backspace') { e.preventDefault(); removeBlock(b.id); }
      else if (e.altKey && e.key === 'ArrowUp' && i > 0) { e.preventDefault(); moveBlock(b.id, i - 1); refocus(b.id); }
      else if (e.altKey && e.key === 'ArrowDown' && i < blocks.length - 1) { e.preventDefault(); moveBlock(b.id, i + 2); refocus(b.id); }
      else if (e.key === 'ArrowDown' && blocks[i + 1]) { e.preventDefault(); focusBlock(blocks[i + 1].id); }
      else if (e.key === 'ArrowUp' && blocks[i - 1]) { e.preventDefault(); focusBlock(blocks[i - 1].id); }
    },
    ondragstart: (e) => { dragInfo = { from: 'script', id: b.id }; e.dataTransfer.setData('text/plain', 'ab:' + b.id); e.dataTransfer.effectAllowed = 'move'; setTimeout(() => e.target.classList && e.target.classList.add('dragging'), 0); },
    ondragend: (e) => { dragInfo = null; e.target.classList && e.target.classList.remove('dragging'); clearDropzones(); },
    // Dropping on a block inserts before (upper half) or after (lower half) it.
    ondragover: (e) => {
      if (!dragInfo) return;
      e.preventDefault();
      const r = e.currentTarget.getBoundingClientRect();
      const top = e.clientY < r.top + r.height / 2;
      clearDropzones();
      const dz = top ? e.currentTarget.previousElementSibling : e.currentTarget.nextElementSibling;
      if (dz && dz.classList.contains('dz')) dz.classList.add('over');
    },
    ondrop: (e) => {
      if (!dragInfo) return;
      const r = e.currentTarget.getBoundingClientRect();
      handleDrop(e.clientY < r.top + r.height / 2 ? i : i + 1, e);
    }
  },
  h('div', { class: 'parts' }, parts),
  info.out ? h('span', { class: 'outlabel' }, info.out.name + ' ' + fmtShape(info.out.shape)) : null,
  bad ? h('span', { class: 'badge', title: info.errors.map((er) => t(er.key, er.args)).join('\n'), 'aria-label': info.errors.length + ' errors' }, '!') : null,
  h('div', { class: 'block-tools' },
    tool('up', t('btn.up'), () => { moveBlock(b.id, i - 1); refocus(b.id); }, i === 0),
    tool('down', t('btn.down'), () => { moveBlock(b.id, i + 2); refocus(b.id); }, i === blocks.length - 1),
    tool('trash', t('btn.delete'), () => removeBlock(b.id))));
  return el;
}

function focusBlock(id) {
  const el = els.canvas.querySelector('.block[data-id="' + id + '"]');
  if (el) el.focus();
}
function refocus(id) { setTimeout(() => focusBlock(id), 0); }

function renderCanvas() {
  const scroll = els.canvas.scrollTop;
  if (state.view === 'diagram') {
    const d = attentionDiagram(state.program, analysis, { showShapes: state.showShapes });
    const card = h('div', { class: 'diagram-card' });
    card.innerHTML = d.svg;
    els.canvas.replaceChildren(h('div', { class: 'diagram-wrap' }, card));
    state.lastFigure = { ...d, name: state.program.name };
    return;
  }
  const blocks = state.program.blocks;
  const script = h('div', { class: 'script' });
  if (!blocks.length) {
    script.append(dropzone(0), h('div', { class: 'empty' }, t('script.empty')));
  } else {
    blocks.forEach((b, i) => { script.append(dropzone(i), blockEl(b, i)); });
    script.append(dropzone(blocks.length));
  }
  els.canvas.replaceChildren(script);
  els.canvas.scrollTop = scroll;
}

async function exportDiagram(kind) {
  const fig = state.lastFigure;
  if (!fig) return;
  downloadFigure(fig, state.program.name, kind);
}

async function downloadFigure(fig, name, kind) {
  if (kind === 'svg') {
    downloadBlob(new Blob([fig.svg], { type: 'image/svg+xml' }), safeName(name) + '.svg');
    toast(t('toast.svg'));
  } else {
    try {
      const blob = await svgToPngBlob(fig.svg, fig.width, fig.height, state.pngScale);
      downloadBlob(blob, safeName(name) + '.png');
      toast(t('toast.png'));
    } catch (e) { toast('PNG export failed'); }
  }
}

// ---------------------------------------------------------------------------
// Inspector
// ---------------------------------------------------------------------------
function renderInspector() {
  const tabs = [['props', 'tab.props'], ['code', 'tab.code'], ['shapes', 'tab.shapes'], ['preview', 'tab.preview']];
  els.inspTabs.replaceChildren(...tabs.map(([id, key]) => h('button', {
    class: 'tab', role: 'tab', 'aria-selected': state.tab === id, onclick: () => { state.tab = id; renderInspector(); }
  }, t(key))));
  const scroll = els.inspBody.scrollTop;
  els.inspBody.replaceChildren();
  if (state.tab === 'props') renderProps();
  else if (state.tab === 'code') renderCode();
  else if (state.tab === 'shapes') renderShapes();
  else renderPreview();
  els.inspBody.scrollTop = scroll;
}

function errList(items, cls) {
  return h('ul', { class: 'errs' }, items.map((er) => h('li', { class: cls || '' }, t(er.key, er.args))));
}

function fieldRow(b, f, info) {
  const val = b.params[f.key];
  const label = t(b.type === 'mask' && f.key === 'k' ? 'f.k_topk' : 'f.' + f.key);
  const set = (v) => commit((p) => { p.blocks.find((x) => x.id === b.id).params[f.key] = v; });
  let control;
  if (f.kind === 'tensor') {
    const opts = Array.from(new Set(info.scope));
    const undef = !opts.includes(val);
    if (undef) opts.push(val);
    control = h('select', { 'aria-label': label, class: undef ? 'undef' : '', onchange: (e) => set(e.target.value) },
      opts.map((o) => h('option', { value: o, selected: o === val }, o + (o === val && undef ? ' ⚠' : ''))));
  } else if (f.kind === 'select') {
    control = h('select', { 'aria-label': label, onchange: (e) => set(e.target.value) }, f.options.map((o) => h('option', { value: o, selected: o === val }, o)));
  } else if (f.kind === 'number') {
    control = h('input', { type: 'number', 'aria-label': label, value: val, min: f.min, max: f.max, step: f.step || 'any', onchange: (e) => { const n = Number(e.target.value); if (Number.isFinite(n)) set(n); else renderInspector(); } });
  } else if (f.kind === 'textarea') {
    control = h('textarea', { 'aria-label': label, spellcheck: 'false', onchange: (e) => set(e.target.value) });
    control.value = val;
  } else {
    control = h('input', { type: 'text', 'aria-label': label, value: val, spellcheck: 'false', onchange: (e) => set(e.target.value.trim()) });
  }
  return h('label', { class: 'field' + (f.kind === 'textarea' ? ' block-field' : '') }, h('span', {}, label), control);
}

function renderProps() {
  const b = selBlock();
  if (!b) {
    add(els.inspBody, 
      h('div', {}, h('h2', {}, t('sec.config')), h('p', {}, t('palette.hint'))),
      configFields(),
      analysis.errors.length ? errList(analysis.errors.filter((e) => e.blockId === null)) : null
    );
    return;
  }
  const def = BLOCKS[b.type];
  const cat = CATEGORIES[def.cat];
  const info = analysis.perBlock[b.id];
  const sec = h('div', { class: 'sec' }, h('h3', {}, t('sec.params')),
    ...def.fields.filter((f) => !f.when || f.when(b.params)).map((f) => fieldRow(b, f, info)));
  els.inspBody.style.setProperty('--c', cat.color);
  els.inspBody.style.setProperty('--t', cat.tint);
  add(els.inspBody, 
    h('div', {}, h('span', { class: 'cat-chip' }, t('cat.' + def.cat)), h('h2', {}, t('block.' + b.type + '.name')), h('p', {}, t('block.' + b.type + '.desc'))),
    info.errors.length ? errList(info.errors) : null,
    sec
  );
  if (def.alts) {
    add(els.inspBody, h('div', { class: 'sec' }, h('h3', {}, t('sec.alts')),
      h('div', { class: 'alts' }, def.alts.map((a) => h('button', { class: 'alt', 'aria-pressed': b.params.fn === a, onclick: () => commit((p) => { p.blocks.find((x) => x.id === b.id).params.fn = a; }) }, a)))));
  }
  const ins = info.ins.filter((x, i, arr) => arr.findIndex((y) => y.name === x.name) === i).map((x) => x.name + ' ' + fmtShape(x.shape)).join(', ') || '—';
  add(els.inspBody, h('div', { class: 'iobox' },
    h('div', {}, h('span', {}, t('io.in')), h('span', {}, ins)),
    h('div', {}, h('span', {}, t('io.out')), h('span', {}, info.out ? info.out.name + ' ' + fmtShape(info.out.shape) : '—')),
    h('div', {}, h('span', {}, t('io.cost')), h('span', {}, info.cost)),
    info.params ? h('div', {}, h('span', {}, t('io.params')), h('span', {}, info.params.toLocaleString('en-US'))) : null));
}

function configFields() {
  const cfg = state.program.config;
  return h('div', { class: 'sec' }, ...Object.keys(DEFAULT_CONFIG).map((k) => h('label', { class: 'field' }, h('span', {}, t('cfg.' + k)),
    h('input', { type: 'number', min: 1, step: 1, value: cfg[k], 'aria-label': t('cfg.' + k), onchange: (e) => { const n = Math.floor(Number(e.target.value)); if (n >= 1) commit((p) => { p.config[k] = n; }); else renderInspector(); } }))));
}

function renderCode() {
  const gen = generateCode(state.program, analysis);
  const lines = gen.lines.map((l, i) => h('div', { class: 'ln' + (l.blockId && l.blockId === state.selId ? ' hl' : '') },
    h('span', { class: 'n' }, i + 1), h('span', { class: 'tx' }, l.text || ' ')));
  add(els.inspBody, 
    h('div', { class: 'row between' }, h('span', { class: 'muted' }, 'PyTorch'),
      h('div', { class: 'row' },
        h('button', { class: 'btn small', onclick: async () => { try { await navigator.clipboard.writeText(gen.text); toast(t('toast.copied')); } catch (e) { /* ignore */ } } }, icon('copy', 14), t('btn.copy')),
        h('button', { class: 'btn small', onclick: () => downloadBlob(new Blob([gen.text], { type: 'text/x-python' }), safeName(state.program.name) + '.py') }, icon('download', 14), '.py'))),
    analysis.errors.length ? errList(analysis.errors.slice(0, 3)) : null,
    analysis.warnings.length ? errList(analysis.warnings, 'warn') : null,
    h('div', { class: 'code' }, lines));
}

function renderShapes() {
  add(els.inspBody, 
    h('p', { class: 'muted' }, 'B=' + state.program.config.B + ' · T=' + state.program.config.T + ' · D=' + state.program.config.D + ' · H=' + state.program.config.H + ' · h_kv=' + state.program.config.Hkv + ' · Dh=' + (state.program.config.D / state.program.config.H)),
    h('div', { class: 'shape-table' }, analysis.tensors.map((x) => h('div', { class: 'shape-row' + (x.blockId === state.selId ? ' sel' : '') },
      h('span', { class: 'nm' }, x.name), h('span', { class: 'sh' }, fmtShape(x.shape))))),
    analysis.errors.length ? errList(analysis.errors) : null);
}

// Preview (numeric evaluation + heat map) ------------------------------------------------
let evalCache = { key: '', value: null };
function runEval() {
  const key = JSON.stringify(state.program) + '|' + state.seed;
  if (evalCache.key !== key) evalCache = { key, value: analysis.ok ? evaluate(state.program, analysis, state.seed) : null };
  return evalCache.value;
}

function heatColor(v) {
  // white -> blue -> deep navy
  const a = [255, 255, 255], b = [47, 111, 222], c = [11, 42, 102];
  const mix = (p, q, k) => p.map((x, i) => Math.round(x + (q[i] - x) * k));
  const col = v < 0.6 ? mix(a, b, v / 0.6) : mix(b, c, (v - 0.6) / 0.4);
  return col;
}

function drawHeat(canvas, tensor, head) {
  const T = tensor.shape[1];
  canvas.width = T; canvas.height = T;
  const ctx = canvas.getContext('2d');
  const img = ctx.createImageData(T, T);
  let min = Infinity, max = -Infinity;
  for (let i = 0; i < tensor.data.length; i++) { const v = tensor.data[i]; if (Number.isFinite(v)) { min = Math.min(min, v); max = Math.max(max, v); } }
  if (!(max > min)) { max = min + 1; }
  for (let i = 0; i < T; i++) for (let j = 0; j < T; j++) {
    const v = tensor.data[(head * T + i) * T + j];
    const o = (i * T + j) * 4;
    let col;
    if (!Number.isFinite(v)) col = (i + j) % 2 ? [213, 218, 227] : [238, 241, 245];
    else col = heatColor((v - min) / (max - min));
    img.data[o] = col[0]; img.data[o + 1] = col[1]; img.data[o + 2] = col[2]; img.data[o + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function renderPreview() {
  const res = runEval();
  if (!res) { add(els.inspBody, errList([{ key: 'eval.blockerror', args: {} }])); return; }
  const opts = state.program.blocks.filter((b) => isAttentionMap(res.outputs[b.id]));
  const extra = [];
  if (res.stopped) extra.push(errList([{ key: res.stopped.key, args: {} }], 'warn'));
  if (!opts.length) { add(els.inspBody, ...extra, h('p', { class: 'muted' }, t('preview.none'))); return; }
  let target = opts.find((b) => b.id === state.previewBlock);
  if (!target) target = [...opts].reverse().find((b) => b.type === 'norm' || b.type === 'dropout') || opts[opts.length - 1];
  const tensor = res.outputs[target.id];
  const H = tensor.shape[0];
  if (state.previewHead >= H) state.previewHead = 0;
  const canvas = h('canvas', { class: 'canvas-heat', role: 'img', 'aria-label': t('tab.preview') });
  drawHeat(canvas, tensor, state.previewHead);
  const isWeights = target.type === 'norm' || target.type === 'dropout';
  const st = isWeights ? attentionStats({ shape: [1, tensor.shape[1], tensor.shape[2]], data: tensor.data.slice(state.previewHead * tensor.shape[1] * tensor.shape[2], (state.previewHead + 1) * tensor.shape[1] * tensor.shape[2]) }) : null;
  add(els.inspBody, 
    ...extra,
    h('label', { class: 'field block-field' }, t('preview.tensor'),
      h('select', { onchange: (e) => { state.previewBlock = e.target.value; renderInspector(); } },
        opts.map((b) => h('option', { value: b.id, selected: b.id === target.id }, analysis.perBlock[b.id].out.name + ' · ' + t('block.' + b.type + '.name'))))),
    h('div', { class: 'sec' }, h('h3', {}, t('preview.head')),
      h('div', { class: 'seg-heads' }, Array.from({ length: H }, (_, k) => h('button', { 'aria-pressed': k === state.previewHead, onclick: () => { state.previewHead = k; renderInspector(); } }, String(k))))),
    canvas,
    h('div', { class: 'row between muted' }, h('span', {}, t('preview.rows') + '  ' + t('preview.cols')), h('span', { class: 'legend' }, h('i'), h('b'), t('preview.masked'))),
    st ? h('div', { class: 'stats' },
      h('span', {}, t('preview.entropy')), h('span', {}, st.entropy.toFixed(3)),
      h('span', {}, t('preview.sparsity')), h('span', {}, (st.sparsity * 100).toFixed(1) + '%'),
      h('span', {}, t('preview.max')), h('span', {}, st.meanMax.toFixed(3))) : null,
    h('p', { class: 'muted' }, t('preview.note', res.cfg)),
    h('button', { class: 'btn small', onclick: () => { state.seed += 1; renderInspector(); } }, t('btn.reroll')));
}

// ---------------------------------------------------------------------------
// Status bar
// ---------------------------------------------------------------------------
function renderStatus() {
  const errCount = analysis.errors.length;
  const cfg = state.program.config;
  const field = (k) => h('label', { class: 'cfg' }, k, h('input', {
    type: 'number', min: 1, step: 1, value: cfg[k], 'aria-label': t('cfg.' + k),
    onchange: (e) => { const n = Math.floor(Number(e.target.value)); if (n >= 1) commit((p) => { p.config[k] = n; }); else renderStatus(); }
  }));
  els.status.replaceChildren(
    errCount ? h('span', { class: 'bad' }, h('span', { class: 'dotc' }), t('status.err', { n: errCount }))
      : h('span', { class: 'ok' }, h('span', { class: 'dotc' }), t('status.ok')),
    field('B'), field('T'), field('D'), field('H'), field('Hkv'),
    h('span', { class: 'spacer' }),
    h('span', {}, t('status.attn', { n: analysis.attnEntries.toLocaleString('en-US') })),
    h('span', {}, t('status.params', { n: analysis.params >= 1e6 ? (analysis.params / 1e6).toFixed(2) + 'M' : analysis.params.toLocaleString('en-US') })));
}

// ---------------------------------------------------------------------------
// Hugging Face dialog
// ---------------------------------------------------------------------------
const HF_EXAMPLES = ['Qwen/Qwen2.5-7B', 'mistralai/Mixtral-8x7B-v0.1', 'openai-community/gpt2', 'google-bert/bert-base-uncased', 'google-t5/t5-small', 'deepseek-ai/DeepSeek-V2-Lite'];

function openHf() {
  const dlg = document.getElementById('hf-dialog');
  renderHf();
  if (!dlg.open) dlg.showModal();
}

function applySpec(spec) {
  state.hf.spec = spec;
  state.hf.error = null;
  state.hf.note = Math.abs(spec.headDim * spec.heads - spec.hidden) > 0.5 ? 'hf.dimnote' : null;
}

async function hfFetch() {
  state.hf.busy = true; state.hf.error = null; state.hf.spec = null;
  renderHf();
  try {
    const { ref, config } = await fetchHfConfig(state.hf.url);
    applySpec(configToSpec(config, ref.repo));
  } catch (e) {
    state.hf.error = { key: e.code || 'hf.network', args: { missing: (e.missing || []).join(', ') } };
  }
  state.hf.busy = false;
  renderHf();
}

function hfUsePasted() {
  try {
    const cfg = JSON.parse(state.hf.pasted);
    const name = (cfg.architectures && cfg.architectures[0]) || cfg.model_type || 'pasted-config';
    applySpec(configToSpec(cfg, name));
  } catch (e) {
    state.hf.spec = null;
    state.hf.error = { key: e.code || 'hf.badjson', args: { missing: (e.missing || []).join(', ') } };
  }
  renderHf();
}

function renderHf() {
  const dlg = document.getElementById('hf-dialog');
  const s = state.hf.spec;
  const kids = [
    h('div', { class: 'dlg-head' }, h('h2', { id: 'hf-title' }, t('hf.title')),
      h('button', { class: 'btn small', onclick: () => dlg.close() }, t('btn.close'))),
  ];
  const body = h('div', { class: 'dlg-body' },
    h('p', { class: 'muted', style: 'margin:0' }, t('hf.desc')),
    h('form', { class: 'urlrow', onsubmit: (e) => { e.preventDefault(); hfFetch(); } },
      h('input', { type: 'text', 'aria-label': 'Hugging Face URL', placeholder: t('hf.placeholder'), value: state.hf.url, spellcheck: 'false', oninput: (e) => { state.hf.url = e.target.value; } }),
      h('button', { class: 'btn primary', type: 'submit', disabled: state.hf.busy }, t('btn.fetch'))),
    h('div', { class: 'examples' }, t('hf.examples') + ':', HF_EXAMPLES.map((x) => h('button', { type: 'button', onclick: () => { state.hf.url = 'https://huggingface.co/' + x; hfFetch(); } }, x))),
    h('details', {}, h('summary', {}, t('hf.paste')),
      (() => { const ta = h('textarea', { 'aria-label': 'config.json', spellcheck: 'false', oninput: (e) => { state.hf.pasted = e.target.value; } }); ta.value = state.hf.pasted; return ta; })(),
      h('button', { class: 'btn small', style: 'margin-top:8px', onclick: hfUsePasted }, t('btn.use_pasted')))
  );
  if (state.hf.busy) body.append(h('div', { class: 'msg-info' }, t('hf.loading')));
  if (state.hf.error) body.append(h('div', { class: 'msg-err', role: 'alert' }, t(state.hf.error.key, state.hf.error.args)));
  if (s) {
    const fig = modelDiagram(s);
    state.hf.figure = fig;
    const attn = s.attnKind === 'MLA' ? 'MLA' : s.attnKind + ' (h=' + s.heads + (s.attnKind !== 'MHA' ? ', h_kv=' + s.kvHeads : '') + ', d_h=' + Math.round(s.headDim) + ')';
    const ffn = s.ffnType === 'moe' ? 'MoE (E=' + s.experts + (s.topK ? ', top-' + s.topK : '') + ')' : (s.ffnType === 'gated' ? 'Gated ' : '') + s.act + ', d_ff=' + s.ffn.toLocaleString('en-US');
    const dl = (k, v) => [h('div', {}, h('dt', {}, t('hf.field.' + k)), h('dd', {}, v))];
    const figBox = h('div', { class: 'fig-box' });
    figBox.innerHTML = fig.svg;
    add(body, 
      h('h3', { style: 'margin:0;font-size:14px' }, t('hf.summary') + ': ' + s.name),
      h('dl', { class: 'summary' },
        ...dl('arch', s.arch || s.modelType || '—'), ...dl('family', t('hf.family.' + s.family)),
        ...dl('layers', s.family === 'encdec' ? s.encLayers + ' + ' + s.decLayers : String(s.layers)),
        ...dl('hidden', s.hidden.toLocaleString('en-US')), ...dl('heads', attn), ...dl('ffn', ffn),
        ...dl('norm', (s.norm === 'rms' ? 'RMSNorm' : 'LayerNorm') + ', ' + s.placement + '-norm'), ...dl('pos', s.pos + (s.ropeTheta ? ' (θ=' + s.ropeTheta.toLocaleString('en-US') + ')' : '')),
        ...dl('vocab', s.vocab ? s.vocab.toLocaleString('en-US') : '—'), ...dl('params', '≈ ' + formatParams(s.params))),
      state.hf.note ? h('div', { class: 'msg-info' }, t(state.hf.note)) : null,
      h('div', { class: 'row' },
        h('button', { class: 'btn small primary', onclick: () => downloadFigure({ ...fig }, s.name + '-architecture', 'png') }, icon('image', 14), t('btn.png')),
        h('button', { class: 'btn small', onclick: () => downloadFigure({ ...fig }, s.name + '-architecture', 'svg') }, icon('download', 14), t('btn.svg')),
        h('label', { class: 'check' }, t('fig.scale'), h('select', { onchange: (e) => { state.pngScale = Number(e.target.value); } }, [2, 3, 4].map((n) => h('option', { value: n, selected: n === state.pngScale }, n + '×')))),
        h('span', { class: 'spacer', style: 'flex:1' }),
        h('button', { class: 'btn small', onclick: () => { replaceProgram(specToProgram(s)); state.view = 'blocks'; dlg.close(); renderAll(); toast(t('hf.opened')); } }, t('btn.open_editor'))),
      figBox,
      h('p', { class: 'muted', style: 'margin:0' }, t('hf.heuristic'))
    );
  }
  kids.push(body);
  const wrap = h('div', { class: 'dlg' }, ...kids);
  dlg.replaceChildren(wrap);
}

// ---------------------------------------------------------------------------
// Boot
// ---------------------------------------------------------------------------
function loadInitial() {
  if (location.hash.startsWith('#p=')) {
    try { const p = decodeShare(location.hash.slice(3)); setTimeout(() => toast(t('toast.loaded_hash')), 300); return p; } catch (e) { /* fall through */ }
  }
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) return sanitizeProgram(JSON.parse(raw));
  } catch (e) { /* ignore */ }
  return TEMPLATES[0].make();
}

function boot() {
  setLang(detectLang());
  state.program = loadInitial();
  state.selId = state.program.blocks.find((b) => b.type === 'norm')?.id || (state.program.blocks[0] && state.program.blocks[0].id) || null;
  recompute();
  document.getElementById('file-input').addEventListener('change', (e) => { if (e.target.files[0]) importJsonFile(e.target.files[0]); e.target.value = ''; });
  document.addEventListener('keydown', (e) => {
    const tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea' || tag === 'select') return;
    if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'z') { e.preventDefault(); if (e.shiftKey) redo(); else undo(); }
    else if ((e.ctrlKey || e.metaKey) && e.key.toLowerCase() === 'y') { e.preventDefault(); redo(); }
  });
  mount();
  // Expose for debugging / automated tests.
  window.__ab = { state, get analysis() { return analysis; } };
}

boot();
