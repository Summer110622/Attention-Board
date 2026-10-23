import { makeProgram, newBlock } from './blocks.js';

const b = newBlock;

function core({ name, config, rope = false, mask = { kind: 'causal' }, norm = {}, score = {}, proj = {}, dropout = false }) {
  const hk = (config && config.Hkv && config.Hkv !== config.H) ? 'Hkv' : 'H';
  const blocks = [
    b('input'),
    b('proj', { src: 'x', out: 'q', heads: 'H', ...proj }),
    b('proj', { src: 'x', out: 'k', heads: hk, ...proj }),
    b('proj', { src: 'x', out: 'v', heads: hk, ...proj })
  ];
  if (rope) blocks.push(b('rope', { target: 'q' }), b('rope', { target: 'k' }));
  blocks.push(b('score', score), b('mask', { target: 's', ...mask }), b('norm', norm));
  if (dropout) blocks.push(b('dropout'));
  blocks.push(b('mix'), b('outproj'), b('ret'));
  return makeProgram(name, config || {}, blocks);
}

export const TEMPLATES = [
  { id: 'vanilla', make: () => core({ name: 'Vanilla causal attention' }) },
  { id: 'gqa_rope', make: () => core({ name: 'GQA + RoPE (Llama-style)', config: { D: 512, H: 8, Hkv: 2, T: 128 }, rope: true }) },
  { id: 'sliding', make: () => core({ name: 'Sliding-window attention', mask: { kind: 'sliding', window: 32 } }) },
  { id: 'sparsemax', make: () => core({ name: 'Sparsemax attention', norm: { fn: 'sparsemax' } }) },
  { id: 'relu2', make: () => core({ name: 'ReLU² attention', norm: { fn: 'relu2' } }) },
  { id: 'topk', make: () => core({ name: 'Top-k sparse attention', mask: { kind: 'topk', k: 16 } }) },
  { id: 'cosine', make: () => core({ name: 'Cosine attention', score: { kind: 'cosine', scale: 'none' }, norm: { temp: 0.1 } }) },
  { id: 'lowrank', make: () => core({ name: 'Low-rank projections', proj: { kind: 'lowrank', rank: 32 } }) }
];

export function emptyProgram() {
  return makeProgram('Untitled attention', {}, []);
}
