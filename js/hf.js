// Hugging Face integration: parse a model URL, fetch config.json (the Hub serves it with CORS
// enabled) and turn it into an architecture spec that the diagram builder understands.
import { makeProgram, newBlock } from './blocks.js';

const err = (code, extra = {}) => Object.assign(new Error(code), { code }, extra);

export function parseHfUrl(input) {
  let s = String(input || '').trim();
  if (!s) throw err('hf.badurl');
  s = s.replace(/[?#].*$/, '').replace(/^https?:\/\//i, '').replace(/^(www\.)?(huggingface\.co|hf\.co)\//i, '');
  s = s.replace(/\/+$/, '');
  const seg = s.split('/').filter(Boolean);
  if (!seg.length || ['datasets', 'spaces', 'docs', 'models'].includes(seg[0]) && seg[0] !== 'models') throw err('hf.badurl');
  if (seg[0] === 'models') seg.shift();
  const markers = ['tree', 'blob', 'resolve', 'raw', 'blame', 'commits'];
  let repo, rest;
  if (seg.length >= 2 && !markers.includes(seg[1])) { repo = seg[0] + '/' + seg[1]; rest = seg.slice(2); }
  else { repo = seg[0]; rest = seg.slice(1); }
  if (!/^[\w.\-]+(\/[\w.\-]+)?$/.test(repo)) throw err('hf.badurl');
  let rev = 'main';
  if (rest.length >= 2 && markers.includes(rest[0])) rev = decodeURIComponent(rest[1]);
  return { repo, rev };
}

export const hfConfigUrl = ({ repo, rev }) => 'https://huggingface.co/' + repo + '/resolve/' + encodeURIComponent(rev) + '/config.json';

export async function fetchHfConfig(input, fetchImpl = (typeof fetch !== 'undefined' ? fetch : null)) {
  const ref = parseHfUrl(input);
  let res;
  try { res = await fetchImpl(hfConfigUrl(ref)); } catch (e) { throw err('hf.network'); }
  if (res.status === 401 || res.status === 403) throw err('hf.gated');
  if (res.status === 404) throw err('hf.notfound');
  if (!res.ok) throw err('hf.network', { status: res.status });
  let json;
  try { json = await res.json(); } catch (e) { throw err('hf.badjson'); }
  return { ref, config: json };
}

const first = (...vals) => { const v = vals.find((x) => x !== undefined && x !== null); return v === undefined ? null : v; };
const isNum = (v) => typeof v === 'number' && Number.isFinite(v);

const ENCODER_TYPES = ['bert', 'roberta', 'xlm-roberta', 'distilbert', 'electra', 'albert', 'deberta', 'deberta-v2', 'camembert', 'mpnet', 'modernbert', 'ernie', 'nomic_bert'];
const POST_NORM_TYPES = ['bart', 'marian'];
const KNOWN_ACTS = /^(gelu|gelu_new|gelu_fast|gelu_pytorch_tanh|gelu_python|quick_gelu|relu|relu2|silu|swish|swiglu|geglu|tanh|sigmoid|gated-gelu|gated-silu|new_gelu|elu|leaky_relu|squared_relu|relu_squared)$/;
const ROPE_TYPES = /llama|mistral|mixtral|qwen|gemma|phi3|olmo|deepseek|gpt_neox|gptj|falcon|granite|cohere|stablelm|smollm|glm|internlm|yi|modernbert|kimi/;
const LEARNED_TYPES = /^(gpt2|opt|gpt_neo|bert|roberta|xlm-roberta|distilbert|electra|albert|deberta|deberta-v2|camembert|mpnet|ernie|nomic_bert|bart|marian)$/;
const RMS_TYPES = /llama|mistral|mixtral|qwen|gemma|phi3|olmo|deepseek|granite|cohere|kimi/;

// Per-layer kinds ('full' | 'linear' | 'sliding' | 'other' | null) for hybrid models.
// Reads Kimi-style `linear_attn_config.{kda_layers, full_attn_layers}` (1-based) and HF-style `layer_types`.
function detectLayerKinds(c, layers) {
  const lac = c.linear_attn_config;
  if (lac && typeof lac === 'object') {
    const kinds = new Array(layers).fill(null);
    const lists = [['full', lac.full_attn_layers], ['linear', lac.kda_layers || lac.linear_attn_layers || lac.linear_layers]];
    const all = lists.flatMap(([, a]) => (Array.isArray(a) ? a.map(Number) : []));
    const base = all.length && Math.min(...all) === 0 ? 0 : 1;
    lists.forEach(([k, arr]) => { if (Array.isArray(arr)) arr.forEach((i) => { const j = Number(i) - base; if (j >= 0 && j < layers) kinds[j] = k; }); });
    if (kinds.some((k) => k === 'linear')) return { kinds, linearName: lac.kda_layers ? 'KDA' : 'linear attention' };
  }
  if (Array.isArray(c.layer_types) && c.layer_types.length === layers) {
    const kinds = c.layer_types.map((x) => {
      const v = String(x).toLowerCase();
      if (/linear|mamba|ssm|rwkv|delta|kda|lightning|recurrent/.test(v)) return 'linear';
      if (/sliding|local|chunk/.test(v)) return 'sliding';
      if (/full|global|attention/.test(v)) return 'full';
      return 'other';
    });
    if (new Set(kinds).size > 1) return { kinds, linearName: 'linear attention' };
  }
  return null;
}

export function configToSpec(raw, repoName = '') {
  const top = raw || {};
  const c = top.text_config || top.llm_config || top.language_config || top;
  const modelType = String(first(c.model_type, top.model_type, '') || '');
  const arch = ((top.architectures || c.architectures || [])[0]) || '';
  const hidden = first(c.hidden_size, c.n_embd, c.d_model, c.dim);
  const layers = first(c.num_hidden_layers, c.n_layer, c.num_layers, c.n_layers, c.encoder_layers);
  const heads = first(c.num_attention_heads, c.n_head, c.num_heads, c.n_heads, c.encoder_attention_heads);
  const missing = [];
  if (!hidden) missing.push('hidden_size');
  if (!layers) missing.push('num_hidden_layers');
  if (!heads) missing.push('num_attention_heads');
  if (missing.length) throw err('hf.missing', { missing });

  const unknown = [];   // things we could not read from config.json (shown as "unknown", never guessed)
  const notes = [];     // { key, args, en }: known limitations of the figure for this config
  const note = (key, en, args = {}) => notes.push({ key, args, en });

  const kv = first(c.num_key_value_heads, c.num_kv_heads, c.n_head_kv, c.multi_query_group_num, c.multi_query === true ? 1 : undefined, heads);

  // --- Multi-head Latent Attention (DeepSeek-V2/V3, Kimi, ...)
  let mla = null;
  if (c.kv_lora_rank) {
    mla = {
      kvLoraRank: c.kv_lora_rank, qLoraRank: c.q_lora_rank || 0,
      qkNope: first(c.qk_nope_head_dim, null), qkRope: first(c.qk_rope_head_dim, null), vHead: first(c.v_head_dim, null)
    };
    if (mla.qkNope == null || mla.qkRope == null || mla.vHead == null) unknown.push('mla_dims');
  }
  const mlaDim = mla && mla.qkNope != null && mla.qkRope != null ? mla.qkNope + mla.qkRope : undefined;
  const headDim = first(mla ? mlaDim : undefined, c.head_dim, c.kv_channels, c.d_kv, mlaDim, hidden / heads);

  const act = String(first(c.hidden_act, c.hidden_activation, c.activation_function, c.feed_forward_proj, 'gelu'));
  if (!KNOWN_ACTS.test(act.toLowerCase())) note('hf.note.act', 'Activation "' + act + '" is not recognised; its name is shown as given.', { act });

  const experts = first(c.num_local_experts, c.num_experts, c.n_routed_experts, c.moe_num_experts);
  const topK = first(c.num_experts_per_tok, c.num_experts_per_token, c.moe_top_k, c.num_selected_experts, c.top_k);
  const shared = first(c.n_shared_experts, c.num_shared_experts, 0);
  let baseFfn = first(c.intermediate_size, c.n_inner, c.d_ff, c.ffn_dim, c.ffn_hidden_size);
  // Documented defaults of these model classes: n_inner = 4 * n_embd when absent/null.
  if (baseFfn == null && /^(gpt2|gptj|gpt_neo|gpt_bigcode)$/.test(modelType)) baseFfn = 4 * hidden;
  const ffnMoe = first(c.moe_intermediate_size, baseFfn);

  let family = 'decoder';
  if (top.is_encoder_decoder === true || c.is_encoder_decoder === true) family = 'encdec';
  else if (ENCODER_TYPES.includes(modelType)) family = 'encoder';

  const encLayers = family === 'encdec' ? first(c.encoder_layers, c.num_layers, layers) : null;
  const decLayers = family === 'encdec' ? first(c.decoder_layers, c.num_decoder_layers, encLayers) : null;

  // --- positional scheme: only state what the config / a known model family tells us
  const ropeParams = c.rope_parameters && typeof c.rope_parameters === 'object' ? c.rope_parameters : null;
  const ropeScaling = c.rope_scaling && typeof c.rope_scaling === 'object' ? c.rope_scaling : null;
  const hasRope = [c.rope_theta, c.rotary_dim, c.rotary_pct, c.partial_rotary_factor, c.rotary_emb_base, c.rope_scaling, c.rope_parameters, c.qk_rope_head_dim].some((v) => v !== undefined && v !== null);
  let pos;
  if (c.alibi === true || modelType === 'bloom' || modelType === 'mpt') pos = 'alibi';
  else if (modelType.startsWith('t5') || modelType === 'mt5' || modelType === 'umt5') pos = 'relative';
  else if (hasRope || ROPE_TYPES.test(modelType)) pos = 'rope';
  else if (LEARNED_TYPES.test(modelType) || family === 'encoder') pos = 'learned';
  else { pos = 'unknown'; unknown.push('pos'); }
  const ropeTheta = first(c.rope_theta, c.rotary_emb_base, ropeParams && ropeParams.rope_theta, ropeScaling && ropeScaling.rope_theta, null);

  // --- normalisation
  let norm;
  if (c.rms_norm_eps != null || /^(t5|mt5|umt5)/.test(modelType) || RMS_TYPES.test(modelType)) norm = 'rms';
  else if (c.layer_norm_eps != null || c.layer_norm_epsilon != null || c.layernorm_epsilon != null || c.norm_epsilon != null || LEARNED_TYPES.test(modelType)) norm = 'layer';
  else { norm = 'unknown'; unknown.push('norm'); }
  let placement = 'pre';
  if (family === 'encoder' && modelType !== 'modernbert') placement = 'post';
  if (POST_NORM_TYPES.includes(modelType)) placement = 'post';

  let ffnType = 'mlp';
  if (experts && experts > 1) ffnType = 'moe';
  else if (/silu|swish|swiglu/.test(act) || /^gated/.test(act) || /^gemma/.test(modelType)) ffnType = 'gated';
  if (ffnType === 'moe' && !topK) unknown.push('topk');
  const ffn = ffnType === 'moe' ? ffnMoe : baseFfn;
  if (ffn == null) unknown.push('ffn');

  // --- dense layers in an otherwise MoE stack
  let denseLayers = 0, denseFfn = null;
  if (ffnType === 'moe') {
    const k = isNum(c.first_k_dense_replace) ? c.first_k_dense_replace : 0;
    const mlpOnly = Array.isArray(c.mlp_only_layers) ? c.mlp_only_layers.length : 0;
    denseLayers = Math.min(layers, k + mlpOnly);
    if (denseLayers > 0) denseFfn = c.moe_intermediate_size != null && c.intermediate_size != null ? c.intermediate_size : null;
    if (isNum(c.moe_layer_freq) && c.moe_layer_freq > 1) note('hf.note.moe_freq', 'MoE is applied only every ' + c.moe_layer_freq + ' layers; the figure does not show this pattern.', { n: c.moe_layer_freq });
  }

  // --- hybrid layer stacks (linear attention + full attention, local + global, ...)
  let hybrid = null;
  const lk = family === 'decoder' ? detectLayerKinds(c, layers) : null;
  if (lk) {
    const count = (k) => lk.kinds.filter((x) => x === k).length;
    hybrid = {
      kinds: lk.kinds, linearName: lk.linearName,
      full: count('full'), linear: count('linear'), sliding: count('sliding'), other: count('other') + count(null),
      window: isNum(c.sliding_window) ? c.sliding_window : null
    };
    if (hybrid.linear > 0) note('hf.note.linear', hybrid.linearName + ' (linear-attention) layers are only labelled; their internals are not drawn.', { name: hybrid.linearName });
  }
  if (isNum(c.num_nextn_predict_layers) && c.num_nextn_predict_layers > 0) note('hf.note.mtp', 'Multi-token-prediction (MTP) layers are not drawn.', { n: c.num_nextn_predict_layers });

  const win = (!hybrid && typeof c.sliding_window === 'number' && c.use_sliding_window !== false) ? c.sliding_window : null;
  const attnKind = mla ? 'MLA' : (kv === heads ? 'MHA' : (kv === 1 ? 'MQA' : 'GQA'));
  if (!mla && Math.abs(headDim * heads - hidden) > 0.5) note('hf.dimnote', 'head_dim × heads differs from hidden_size; d_model is taken from hidden_size.', {});
  const tie = first(c.tie_word_embeddings, top.tie_word_embeddings); // null = not stated in config.json (not claimed in the figure)
  const vocab = first(c.vocab_size, top.vocab_size, null);

  const spec = {
    name: repoName || modelType || 'model',
    modelType, arch, family,
    layers: family === 'encdec' ? encLayers : layers, encLayers, decLayers,
    hidden, heads, kvHeads: kv, headDim, ffn,
    vocab, act, ffnType, experts: experts || null, topK: topK || null, shared: shared || 0,
    denseLayers, denseFfn, hybrid,
    norm, placement, pos, ropeTheta,
    maxPos: first(c.max_position_embeddings, c.n_positions, c.n_ctx, c.max_seq_len, null),
    tie, window: win, mla, attnKind,
    unknown, notes,
    multimodal: top.text_config ? (top.model_type || 'multimodal') : null
  };
  spec.params = estimateParams(spec);
  return spec;
}

// Rough parameter estimate; returns null whenever a needed quantity is unknown or not modelled.
export function estimateParams(s) {
  if (!s.ffn || (s.hybrid && s.hybrid.linear > 0) || (s.ffnType === 'moe' && !s.experts)) return null;
  if (s.mla && (s.mla.qkNope == null || s.mla.qkRope == null || s.mla.vHead == null)) return null;
  const d = s.hidden, hd = s.headDim;
  let attn;
  if (s.mla) {
    const m = s.mla, dq = m.qkNope + m.qkRope;
    const q = m.qLoraRank ? d * m.qLoraRank + m.qLoraRank * s.heads * dq : d * s.heads * dq;
    const kvp = d * (m.kvLoraRank + m.qkRope) + m.kvLoraRank * s.heads * (m.qkNope + m.vHead);
    attn = q + kvp + s.heads * m.vHead * d;
  } else attn = d * s.heads * hd * 2 + 2 * d * s.kvHeads * hd;
  const ffnOf = (type, f) => {
    if (type === 'moe') return (s.experts + (s.shared || 0)) * 3 * d * f + d * s.experts;
    if (type === 'gated') return 3 * d * f;
    return 2 * d * f;
  };
  let total;
  if (s.family === 'encdec') {
    total = s.encLayers * (attn + ffnOf(s.ffnType, s.ffn)) + s.decLayers * (2 * attn + ffnOf(s.ffnType, s.ffn));
  } else if (s.ffnType === 'moe' && s.denseLayers > 0) {
    if (!s.denseFfn) return null;
    total = s.layers * attn + (s.layers - s.denseLayers) * ffnOf('moe', s.ffn) + s.denseLayers * ffnOf('gated', s.denseFfn);
  } else total = s.layers * (attn + ffnOf(s.ffnType, s.ffn));
  if (s.vocab) total += s.vocab * d * ((s.tie !== false || s.family === 'encoder') ? 1 : 2);
  if (s.pos === 'learned' && s.maxPos) total += s.maxPos * d;
  return Math.round(total);
}

export function formatParams(n) {
  if (!n) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + 'M';
  return (n / 1e3).toFixed(0) + 'K';
}

// Build an editable attention program that mirrors the spec's (full / softmax) attention layer.
export function specToProgram(spec) {
  const H = spec.heads, Hkv = spec.kvHeads;
  const D = spec.hidden;
  const dh = Math.round(spec.headDim);
  const explicitDh = !(D % H === 0 && dh * H === D);
  const config = { B: 1, T: 128, D, H, Hkv, Dh: explicitDh ? dh : 0 };
  const m = spec.mla;
  const mlaOk = m && m.qkNope != null && m.qkRope != null && m.vHead != null;
  const blocks = [newBlock('input')];
  if (mlaOk) {
    config.Hkv = H;
    blocks.push(newBlock('mla', { src: 'x', qRank: m.qLoraRank || 0, kvRank: m.kvLoraRank, nope: m.qkNope, rope: m.qkRope, vDim: m.vHead, base: spec.ropeTheta || 10000 }));
  } else {
    const hk = Hkv !== H ? 'Hkv' : 'H';
    blocks.push(
      newBlock('proj', { src: 'x', out: 'q', heads: 'H' }),
      newBlock('proj', { src: 'x', out: 'k', heads: hk }),
      newBlock('proj', { src: 'x', out: 'v', heads: hk })
    );
    if (spec.pos === 'rope') {
      const base = spec.ropeTheta || 10000;
      blocks.push(newBlock('rope', { target: 'q', base }), newBlock('rope', { target: 'k', base }));
    }
  }
  blocks.push(newBlock('score'));
  if (spec.family !== 'encoder') {
    const w = spec.window || (spec.hybrid && spec.hybrid.full === 0 && spec.hybrid.window) || null;
    blocks.push(w ? newBlock('mask', { kind: 'sliding', window: w }) : newBlock('mask', { kind: 'causal' }));
  }
  blocks.push(newBlock('norm'), newBlock('mix'), newBlock('outproj'), newBlock('ret'));
  return makeProgram(spec.name + (spec.hybrid && (spec.hybrid.linear || spec.hybrid.sliding) ? ' (full-attention layer)' : ' (attention)'), config, blocks);
}
