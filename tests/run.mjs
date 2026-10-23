// Dependency-free tests: node tests/run.mjs
import assert from 'node:assert/strict';
import { analyze, generateCode, newBlock, makeProgram } from '../js/blocks.js';
import { TEMPLATES } from '../js/templates.js';
import { evaluate, attentionStats } from '../js/tensor.js';
import { parseHfUrl, configToSpec, specToProgram, fetchHfConfig, formatParams } from '../js/hf.js';
import { attentionDiagram, modelDiagram } from '../js/diagram.js';
import { BLOCKS, PALETTE } from '../js/blocks.js';
import { t, setLang, EN_KEYS, JA_KEYS } from '../js/i18n.js';
import { sanitizeProgram, encodeShare, decodeShare } from '../js/storage.js';

let passed = 0;
const test = (name, fn) => {
  try { fn(); passed++; console.log('ok   ' + name); } catch (e) { console.error('FAIL ' + name + '\n' + (e.stack || e)); process.exitCode = 1; }
};
const testAsync = async (name, fn) => {
  try { await fn(); passed++; console.log('ok   ' + name); } catch (e) { console.error('FAIL ' + name + '\n' + (e.stack || e)); process.exitCode = 1; }
};

// --- templates analyse cleanly, generate code and evaluate ---------------------------------
for (const tpl of TEMPLATES) {
  test('template ' + tpl.id + ': shapes, codegen, eval', () => {
    const p = tpl.make();
    const a = analyze(p);
    assert.deepEqual(a.errors, [], JSON.stringify(a.errors));
    assert.equal(a.warnings.length, 0);
    const code = generateCode(p, a).text;
    assert.match(code, /class CustomAttention\(nn\.Module\)/);
    assert.match(code, /return y/);
    const r = evaluate(p, a);
    assert.equal(r.stopped, null);
    const normBlock = p.blocks.find((b) => b.type === 'norm');
    const attn = r.outputs[normBlock.id];
    const st = attentionStats(attn);
    assert.ok(Number.isFinite(st.entropy));
    if (normBlock.params.fn === 'softmax' || normBlock.params.fn === 'sparsemax' || normBlock.params.fn === 'relu2') {
      const T = attn.shape[1];
      for (let r0 = 0; r0 < attn.shape[0] * T; r0++) {
        let sum = 0;
        for (let j = 0; j < T; j++) sum += attn.data[r0 * T + j];
        // relu2 attention can legitimately produce all-zero rows when every score is negative
        const okSum = Math.abs(sum - 1) < (normBlock.params.fn === 'relu2' ? 2e-3 : 1e-4) || (normBlock.params.fn === 'relu2' && sum === 0);
        assert.ok(okSum, 'row sums to 1, got ' + sum);
      }
    }
  });
}

test('causal mask zeroes future positions', () => {
  const p = TEMPLATES[0].make();
  const a = analyze(p);
  const r = evaluate(p, a);
  const attn = r.outputs[p.blocks.find((b) => b.type === 'norm').id];
  const T = attn.shape[1];
  for (let h = 0; h < attn.shape[0]; h++) for (let i = 0; i < T; i++) for (let j = i + 1; j < T; j++) assert.equal(attn.data[(h * T + i) * T + j], 0);
});

test('sliding window limits attention span', () => {
  const p = TEMPLATES.find((t) => t.id === 'sliding').make();
  const a = analyze(p);
  const r = evaluate(p, a);
  const attn = r.outputs[p.blocks.find((b) => b.type === 'norm').id];
  const T = attn.shape[1];
  const win = Math.max(1, Math.round(32 * r.cfg.scaleT));
  for (let i = 0; i < T; i++) for (let j = 0; j < i - win + 1; j++) assert.equal(attn.data[i * T + j], 0);
});

test('sparsemax output is sparser than softmax', () => {
  const sm = TEMPLATES[0].make(), sp = TEMPLATES.find((t) => t.id === 'sparsemax').make();
  const stat = (p) => attentionStats(evaluate(p, analyze(p)).outputs[p.blocks.find((b) => b.type === 'norm').id]);
  assert.ok(stat(sp).sparsity > stat(sm).sparsity);
});

test('GQA template: k/v use Hkv, codegen repeats kv', () => {
  const p = TEMPLATES.find((t) => t.id === 'gqa_rope').make();
  const a = analyze(p);
  assert.deepEqual(a.errors, []);
  const code = generateCode(p, a).text;
  assert.match(code, /repeat_kv\(k, H \/\/ Hkv\)/);
  assert.match(code, /apply_rope\(q/);
  assert.match(code, /n_kv_heads \* self\.Dh/);
  const kProj = p.blocks.find((b) => b.type === 'proj' && b.params.out === 'k');
  assert.deepEqual(a.perBlock[kProj.id].out.shape, ['B', 'Hkv', 'T', 'Dh']);
});

test('generated python is syntactically valid-looking (balanced parens per line)', () => {
  for (const tpl of TEMPLATES) {
    const p = tpl.make();
    generateCode(p, analyze(p)).text.split('\n').forEach((ln) => {
      const code = ln.replace(/#.*$/, '').replace(/"[^"]*"/g, '');
      assert.equal((code.match(/\(/g) || []).length, (code.match(/\)/g) || []).length, ln);
      assert.equal((code.match(/\[/g) || []).length, (code.match(/\]/g) || []).length, ln);
    });
  }
});

// --- error detection ----------------------------------------------------------------------
test('undefined tensor is reported on the right block', () => {
  const p = makeProgram('t', {}, [newBlock('input'), newBlock('score', { q: 'nope' })]);
  const a = analyze(p);
  assert.ok(a.errors.some((e) => e.key === 'err.undefined' && e.args.name === 'nope'));
});

test('shape mismatch: normalising x (3-D) as scores', () => {
  const p = makeProgram('t', {}, [newBlock('input'), newBlock('norm', { target: 'x' })]);
  assert.ok(analyze(p).errors.some((e) => e.key === 'err.shape'));
});

test('invalid config: D not divisible by H, GQA group mismatch', () => {
  const p = makeProgram('t', { D: 100, H: 8, Hkv: 3 }, []);
  const keys = analyze(p).errors.map((e) => e.key);
  assert.ok(keys.includes('err.cfg_dh'));
  assert.ok(keys.includes('err.gqa'));
});

test('custom block participates in shape inference and stops evaluation', () => {
  const p = TEMPLATES[0].make();
  const idx = p.blocks.findIndex((b) => b.type === 'norm');
  p.blocks.splice(idx, 0, newBlock('custom', { inputs: 's', outName: 's', outShape: 'B, H, T, T', code: 's = s * 2' }));
  const a = analyze(p);
  assert.deepEqual(a.errors, []);
  assert.match(generateCode(p, a).text, /s = s \* 2/);
  assert.equal(evaluate(p, a).stopped.key, 'eval.custom');
});

test('parameter count matches 4 * D^2 for vanilla MHA', () => {
  const a = analyze(TEMPLATES[0].make());
  assert.equal(a.params, 4 * 512 * 512);
});

// --- Hugging Face ---------------------------------------------------------------------------
test('parseHfUrl variants', () => {
  assert.deepEqual(parseHfUrl('https://huggingface.co/meta-llama/Llama-3.1-8B'), { repo: 'meta-llama/Llama-3.1-8B', rev: 'main' });
  assert.deepEqual(parseHfUrl('https://huggingface.co/mistralai/Mistral-7B-v0.1/blob/main/config.json'), { repo: 'mistralai/Mistral-7B-v0.1', rev: 'main' });
  assert.deepEqual(parseHfUrl('https://huggingface.co/Qwen/Qwen2.5-7B/tree/refs%2Fpr%2F3?x=1'), { repo: 'Qwen/Qwen2.5-7B', rev: 'refs/pr/3' });
  assert.deepEqual(parseHfUrl('hf.co/gpt2'), { repo: 'gpt2', rev: 'main' });
  assert.deepEqual(parseHfUrl('openai-community/gpt2'), { repo: 'openai-community/gpt2', rev: 'main' });
  assert.throws(() => parseHfUrl('https://huggingface.co/datasets/foo/bar'), /hf.badurl/);
  assert.throws(() => parseHfUrl(''), /hf.badurl/);
});

const LLAMA3_8B = {
  architectures: ['LlamaForCausalLM'], model_type: 'llama', hidden_size: 4096, intermediate_size: 14336, num_attention_heads: 32,
  num_hidden_layers: 32, num_key_value_heads: 8, hidden_act: 'silu', rms_norm_eps: 1e-5, rope_theta: 500000.0, vocab_size: 128256,
  max_position_embeddings: 8192, tie_word_embeddings: false
};
const GPT2 = { architectures: ['GPT2LMHeadModel'], model_type: 'gpt2', n_embd: 768, n_layer: 12, n_head: 12, n_positions: 1024, vocab_size: 50257, activation_function: 'gelu_new', layer_norm_epsilon: 1e-5 };
const BERT = { architectures: ['BertForMaskedLM'], model_type: 'bert', hidden_size: 768, num_hidden_layers: 12, num_attention_heads: 12, intermediate_size: 3072, hidden_act: 'gelu', vocab_size: 30522, max_position_embeddings: 512 };
const T5 = { architectures: ['T5ForConditionalGeneration'], model_type: 't5', d_model: 512, d_ff: 2048, num_layers: 6, num_decoder_layers: 6, num_heads: 8, d_kv: 64, vocab_size: 32128, is_encoder_decoder: true, feed_forward_proj: 'relu' };
const MIXTRAL = { architectures: ['MixtralForCausalLM'], model_type: 'mixtral', hidden_size: 4096, intermediate_size: 14336, num_attention_heads: 32, num_hidden_layers: 32, num_key_value_heads: 8, num_local_experts: 8, num_experts_per_tok: 2, hidden_act: 'silu', rms_norm_eps: 1e-5, rope_theta: 1e6, vocab_size: 32000, sliding_window: null };
const MULTIMODAL = { model_type: 'llava', text_config: { model_type: 'llama', hidden_size: 4096, intermediate_size: 11008, num_attention_heads: 32, num_hidden_layers: 32, hidden_act: 'silu', rms_norm_eps: 1e-6, vocab_size: 32064 } };

test('configToSpec: Llama-3-8B', () => {
  const s = configToSpec(LLAMA3_8B, 'meta-llama/Llama-3-8B');
  assert.equal(s.family, 'decoder'); assert.equal(s.attnKind, 'GQA'); assert.equal(s.ffnType, 'gated');
  assert.equal(s.norm, 'rms'); assert.equal(s.pos, 'rope'); assert.equal(s.placement, 'pre'); assert.equal(s.headDim, 128);
  assert.equal(formatParams(s.params), '8.0B');
});
test('configToSpec: GPT-2 (learned pos, LayerNorm, MLP, MHA)', () => {
  const s = configToSpec(GPT2, 'gpt2');
  assert.equal(s.attnKind, 'MHA'); assert.equal(s.pos, 'learned'); assert.equal(s.norm, 'layer'); assert.equal(s.ffnType, 'mlp'); assert.equal(s.ffn, 3072);
  assert.equal(formatParams(s.params), '124M');
});
test('configToSpec: BERT is encoder-only post-norm', () => {
  const s = configToSpec(BERT, 'bert-base');
  assert.equal(s.family, 'encoder'); assert.equal(s.placement, 'post');
  assert.equal(formatParams(s.params), '109M');
});
test('configToSpec: T5 is encoder-decoder with relative positions', () => {
  const s = configToSpec(T5, 't5-small');
  assert.equal(s.family, 'encdec'); assert.equal(s.pos, 'relative'); assert.equal(s.norm, 'rms'); assert.equal(s.decLayers, 6);
});
test('configToSpec: Mixtral is MoE', () => {
  const s = configToSpec(MIXTRAL, 'mixtral');
  assert.equal(s.ffnType, 'moe'); assert.equal(s.experts, 8); assert.equal(s.topK, 2);
});
test('configToSpec: nested text_config', () => {
  const s = configToSpec(MULTIMODAL, 'llava');
  assert.equal(s.hidden, 4096); assert.equal(s.multimodal, 'llava');
});
test('configToSpec: missing fields -> hf.missing', () => {
  assert.throws(() => configToSpec({ model_type: 'foo' }), (e) => e.code === 'hf.missing' && e.missing.includes('hidden_size'));
});
test('specToProgram yields an error-free attention program', () => {
  for (const cfg of [LLAMA3_8B, GPT2, BERT, MIXTRAL]) {
    const p = specToProgram(configToSpec(cfg, 'x'));
    const a = analyze(p);
    assert.deepEqual(a.errors, [], JSON.stringify(a.errors));
  }
});

await testAsync('fetchHfConfig uses resolve URL and maps errors', async () => {
  const calls = [];
  const ok = async (u) => { calls.push(u); return { ok: true, status: 200, json: async () => GPT2 }; };
  const r = await fetchHfConfig('https://huggingface.co/openai-community/gpt2', ok);
  assert.equal(calls[0], 'https://huggingface.co/openai-community/gpt2/resolve/main/config.json');
  assert.equal(r.config.model_type, 'gpt2');
  await assert.rejects(() => fetchHfConfig('foo/bar', async () => ({ ok: false, status: 404 })), /hf.notfound/);
  await assert.rejects(() => fetchHfConfig('foo/bar', async () => ({ ok: false, status: 401 })), /hf.gated/);
  await assert.rejects(() => fetchHfConfig('foo/bar', async () => { throw new TypeError('x'); }), /hf.network/);
});

// --- diagrams -------------------------------------------------------------------------------
const wellFormed = (svg) => {
  assert.match(svg, /^<svg [^>]*xmlns="http:\/\/www.w3.org\/2000\/svg"/);
  assert.ok(svg.endsWith('</svg>'));
  assert.ok(!/NaN|undefined/.test(svg), 'no NaN/undefined in svg');
  const opens = (svg.match(/<text /g) || []).length, closes = (svg.match(/<\/text>/g) || []).length;
  assert.equal(opens, closes);
};
test('attention diagrams for all templates', () => {
  for (const tpl of TEMPLATES) {
    const p = tpl.make();
    const d = attentionDiagram(p, analyze(p));
    wellFormed(d.svg);
    assert.ok(d.width > 300 && d.height > 300);
  }
});
test('model diagrams for decoder / encoder / enc-dec / MoE / multimodal', () => {
  for (const [cfg, name] of [[LLAMA3_8B, 'Llama-3-8B'], [GPT2, 'GPT-2'], [BERT, 'BERT'], [T5, 'T5'], [MIXTRAL, 'Mixtral'], [MULTIMODAL, 'LLaVA']]) {
    const d = modelDiagram(configToSpec(cfg, name));
    wellFormed(d.svg);
    assert.ok(d.svg.includes(name));
  }
});
test('empty program renders a placeholder', () => {
  const p = makeProgram('e', {}, []);
  wellFormed(attentionDiagram(p, analyze(p)).svg);
});

// --- i18n -----------------------------------------------------------------------------------
test('every Japanese key also exists in English', () => {
  const en = new Set(EN_KEYS);
  JA_KEYS.forEach((k) => assert.ok(en.has(k), 'missing in EN: ' + k));
});
test('all block names, descriptions, field labels, categories and templates are translated', () => {
  const need = [];
  Object.entries(BLOCKS).forEach(([type, def]) => {
    need.push('block.' + type + '.name', 'block.' + type + '.desc');
    def.fields.forEach((f) => need.push(type === 'mask' && f.key === 'k' ? 'f.k_topk' : 'f.' + f.key));
  });
  PALETTE.forEach((g) => need.push('cat.' + g.cat));
  TEMPLATES.forEach((tp) => need.push('tpl.' + tp.id));
  ['B', 'T', 'D', 'H', 'Hkv'].forEach((k) => need.push('cfg.' + k));
  const en = new Set(EN_KEYS), ja = new Set(JA_KEYS);
  need.forEach((k) => { assert.ok(en.has(k), 'EN missing ' + k); if (k !== 'f.k_topk') assert.ok(ja.has(k), 'JA missing ' + k); });
});
test('every error/warning key produced by the checker is translated', () => {
  const keys = new Set();
  const progs = [
    makeProgram('x', { D: 100, H: 8, Hkv: 3, B: 0 }, []),
    makeProgram('x', {}, [newBlock('score', { q: 'zz' }), newBlock('norm', { target: 'x', out: '1bad', temp: 0 }), newBlock('mask', { kind: 'topk', k: 9999, target: 'x' }), newBlock('dropout', { p: 2, target: 'x' }), newBlock('rope', { target: 'x' }), newBlock('custom', { outShape: '' })])
  ];
  progs.forEach((p) => { const a = analyze(p); a.errors.concat(a.warnings).forEach((e) => keys.add(e.key)); });
  assert.ok(keys.size >= 8, 'collected ' + keys.size + ' keys');
  const en = new Set(EN_KEYS), ja = new Set(JA_KEYS);
  keys.forEach((k) => { assert.ok(en.has(k) && ja.has(k), 'untranslated: ' + k); });
});
test('t() interpolates and switches language', () => {
  setLang('en'); assert.equal(t('err.gqa', { H: 8, Hkv: 3 }), 'H=8 must be divisible by the number of KV heads 3.');
  setLang('ja'); assert.match(t('err.gqa', { H: 8, Hkv: 3 }), /KVヘッド数 3/);
  setLang('en');
});

// --- persistence ----------------------------------------------------------------------------
test('share link round-trips a program (including non-ASCII names)', () => {
  const p = TEMPLATES.find((x) => x.id === 'gqa_rope').make();
  p.name = 'GQA 実験 α';
  const back = decodeShare(encodeShare(p));
  assert.equal(back.name, p.name);
  assert.deepEqual(back.config, p.config);
  assert.deepEqual(back.blocks.map((b) => [b.type, b.params]), p.blocks.map((b) => [b.type, b.params]));
});
test('sanitizeProgram rejects unknown blocks and repairs bad values', () => {
  assert.throws(() => sanitizeProgram({ blocks: [{ type: 'evil' }] }));
  assert.throws(() => sanitizeProgram(null));
  const p = sanitizeProgram({ name: 5, config: { D: -4, H: 'x', T: 64.7 }, blocks: [{ id: 'a', type: 'proj', params: { rank: 'nope', out: 'q' } }, { id: 'a', type: 'ret' }] });
  assert.equal(p.config.D, 512); assert.equal(p.config.T, 64);
  assert.equal(p.blocks[0].params.rank, 64);
  assert.notEqual(p.blocks[0].id, p.blocks[1].id);
});

console.log('\n' + passed + ' passed' + (process.exitCode ? ', with failures' : ''));
