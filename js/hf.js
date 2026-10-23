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

const first = (...vals) => vals.find((v) => v !== undefined && v !== null);

const ENCODER_TYPES = ['bert', 'roberta', 'xlm-roberta', 'distilbert', 'electra', 'albert', 'deberta', 'deberta-v2', 'camembert', 'mpnet', 'modernbert', 'ernie', 'nomic_bert'];
const POST_NORM_TYPES = ['bart', 'marian'];

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

  const kv = first(c.num_key_value_heads, c.num_kv_heads, c.n_head_kv, c.multi_query_group_num, c.multi_query === true ? 1 : undefined, heads);
  const mlaDim = (c.qk_nope_head_dim && c.qk_rope_head_dim) ? c.qk_nope_head_dim + c.qk_rope_head_dim : undefined;
  const headDim = first(c.head_dim, c.kv_channels, c.d_kv, mlaDim, hidden / heads);
  const act = String(first(c.hidden_act, c.hidden_activation, c.activation_function, c.feed_forward_proj, 'gelu'));
  const experts = first(c.num_local_experts, c.num_experts, c.n_routed_experts, c.moe_num_experts);
  const topK = first(c.num_experts_per_tok, c.moe_top_k, c.num_selected_experts, c.top_k);
  const shared = first(c.n_shared_experts, c.num_shared_experts, 0);
  const baseFfn = first(c.intermediate_size, c.n_inner, c.d_ff, c.ffn_dim, c.ffn_hidden_size, 4 * hidden);
  const ffnMoe = first(c.moe_intermediate_size, baseFfn);

  let family = 'decoder';
  if (top.is_encoder_decoder === true || c.is_encoder_decoder === true) family = 'encdec';
  else if (ENCODER_TYPES.includes(modelType)) family = 'encoder';

  const encLayers = family === 'encdec' ? first(c.encoder_layers, c.num_layers, layers) : null;
  const decLayers = family === 'encdec' ? first(c.decoder_layers, c.num_decoder_layers, encLayers) : null;

  const hasRope = [c.rope_theta, c.rotary_dim, c.rotary_pct, c.partial_rotary_factor, c.rotary_emb_base, c.rope_scaling, c.rope_parameters].some((v) => v !== undefined && v !== null);
  let pos = 'learned';
  if (c.alibi === true || modelType === 'bloom' || modelType === 'mpt') pos = 'alibi';
  else if (modelType.startsWith('t5') || modelType === 'mt5' || modelType === 'umt5') pos = 'relative';
  else if (hasRope || /llama|mistral|mixtral|qwen|gemma|phi3|olmo|deepseek|gpt_neox|gptj|falcon|granite|cohere|stablelm|smollm|glm|internlm|yi|modernbert/.test(modelType)) pos = 'rope';
  else if (modelType === 'bart' || family === 'encoder' || /gpt2|opt|gpt_neo$/.test(modelType)) pos = 'learned';

  const norm = (c.rms_norm_eps != null || /^(t5|mt5|umt5)/.test(modelType) || /llama|mistral|qwen|gemma|phi3|olmo|deepseek|granite|cohere/.test(modelType)) ? 'rms' : 'layer';
  let placement = 'pre';
  if (family === 'encoder' && modelType !== 'modernbert') placement = 'post';
  if (POST_NORM_TYPES.includes(modelType)) placement = 'post';

  let ffnType = 'mlp';
  if (experts && experts > 1) ffnType = 'moe';
  else if (/silu|swish|swiglu/.test(act) || /^gated/.test(act) || /^gemma/.test(modelType)) ffnType = 'gated';

  const win = (typeof c.sliding_window === 'number' && c.use_sliding_window !== false) ? c.sliding_window : null;
  const mla = c.kv_lora_rank ? { kvLoraRank: c.kv_lora_rank, qLoraRank: c.q_lora_rank || null } : null;
  const attnKind = mla ? 'MLA' : (kv === heads ? 'MHA' : (kv === 1 ? 'MQA' : 'GQA'));
  const tie = first(c.tie_word_embeddings, top.tie_word_embeddings, true);
  const vocab = first(c.vocab_size, top.vocab_size, null);

  const spec = {
    name: repoName || modelType || 'model',
    modelType, arch, family,
    layers: family === 'encdec' ? encLayers : layers, encLayers, decLayers,
    hidden, heads, kvHeads: kv, headDim, ffn: ffnType === 'moe' ? ffnMoe : baseFfn,
    vocab, act, ffnType, experts: experts || null, topK: topK || null, shared: shared || 0,
    norm, placement, pos, ropeTheta: first(c.rope_theta, c.rotary_emb_base, null),
    maxPos: first(c.max_position_embeddings, c.n_positions, c.n_ctx, c.max_seq_len, null),
    tie, window: win, mla, attnKind,
    multimodal: top.text_config ? (top.model_type || 'multimodal') : null
  };
  spec.params = estimateParams(spec);
  return spec;
}

export function estimateParams(s) {
  const d = s.hidden, hd = s.headDim;
  const attn = s.mla
    ? d * (s.mla.qLoraRank || s.heads * hd) + 4 * d * s.heads * hd
    : d * s.heads * hd * 2 + 2 * d * s.kvHeads * hd;
  let ffn;
  if (s.ffnType === 'moe') ffn = (s.experts + (s.shared || 0)) * 3 * d * s.ffn + d * s.experts;
  else if (s.ffnType === 'gated') ffn = 3 * d * s.ffn;
  else ffn = 2 * d * s.ffn;
  let total;
  if (s.family === 'encdec') {
    total = s.encLayers * (attn + ffn) + s.decLayers * (2 * attn + ffn);
  } else total = s.layers * (attn + ffn);
  if (s.vocab) total += s.vocab * d * ((s.tie || s.family === 'encoder') ? 1 : 2);
  if (s.pos === 'learned' && s.maxPos) total += s.maxPos * d;
  return Math.round(total);
}

export function formatParams(n) {
  if (!n) return '—';
  if (n >= 1e9) return (n / 1e9).toFixed(n >= 1e10 ? 0 : 1) + 'B';
  if (n >= 1e6) return (n / 1e6).toFixed(n >= 1e8 ? 0 : 1) + 'M';
  return (n / 1e3).toFixed(0) + 'K';
}

// Build an editable attention program that mirrors the spec's attention layer.
export function specToProgram(spec) {
  const H = spec.heads, Hkv = spec.kvHeads, Dh = spec.headDim;
  const hk = Hkv !== H ? 'Hkv' : 'H';
  const blocks = [
    newBlock('input'),
    newBlock('proj', { src: 'x', out: 'q', heads: 'H' }),
    newBlock('proj', { src: 'x', out: 'k', heads: hk }),
    newBlock('proj', { src: 'x', out: 'v', heads: hk })
  ];
  if (spec.pos === 'rope') {
    const base = spec.ropeTheta || 10000;
    blocks.push(newBlock('rope', { target: 'q', base }), newBlock('rope', { target: 'k', base }));
  }
  blocks.push(newBlock('score'));
  if (spec.family !== 'encoder') {
    blocks.push(spec.window ? newBlock('mask', { kind: 'sliding', window: spec.window }) : newBlock('mask', { kind: 'causal' }));
  }
  blocks.push(newBlock('norm'), newBlock('mix'), newBlock('outproj'), newBlock('ret'));
  return makeProgram(spec.name + ' (attention)', { B: 1, T: 128, D: Math.round(H * Dh), H, Hkv }, blocks);
}
