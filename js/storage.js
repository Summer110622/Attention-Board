// Persistence helpers: program validation (for imports / share links) and share-link encoding.
import { BLOCKS, DEFAULT_CONFIG, makeProgram } from './blocks.js';

let uid = 0;
const freshId = () => 'b' + Date.now().toString(36) + (uid++).toString(36);

export function sanitizeProgram(obj) {
  if (!obj || typeof obj !== 'object' || !Array.isArray(obj.blocks)) throw new Error('invalid program');
  const config = {};
  Object.keys(DEFAULT_CONFIG).forEach((k) => {
    const v = Number(obj.config && obj.config[k]);
    config[k] = Number.isFinite(v) && v >= 1 ? Math.floor(v) : DEFAULT_CONFIG[k];
  });
  const seen = new Set();
  const blocks = obj.blocks.slice(0, 200).map((b) => {
    if (!b || !BLOCKS[b.type]) throw new Error('unknown block type');
    const defaults = BLOCKS[b.type].defaults();
    const params = {};
    Object.keys(defaults).forEach((k) => {
      const v = b.params ? b.params[k] : undefined;
      if (v === undefined || v === null) params[k] = defaults[k];
      else if (typeof defaults[k] === 'number') params[k] = Number.isFinite(Number(v)) ? Number(v) : defaults[k];
      else params[k] = String(v).slice(0, 4000);
    });
    let id = typeof b.id === 'string' && /^[\w-]{1,40}$/.test(b.id) ? b.id : freshId();
    while (seen.has(id)) id = freshId();
    seen.add(id);
    return { id, type: b.type, params };
  });
  return makeProgram(String(obj.name || 'Untitled attention').slice(0, 120), config, blocks);
}

const toBase64Url = (bytes) => {
  let bin = '';
  bytes.forEach((b) => { bin += String.fromCharCode(b); });
  return btoa(bin).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
};
const fromBase64Url = (s) => {
  const pad = s.length % 4 ? '='.repeat(4 - (s.length % 4)) : '';
  const bin = atob(s.replace(/-/g, '+').replace(/_/g, '/') + pad);
  return Uint8Array.from(bin, (c) => c.charCodeAt(0));
};

export function encodeShare(program) {
  return toBase64Url(new TextEncoder().encode(JSON.stringify(program)));
}
export function decodeShare(str) {
  return sanitizeProgram(JSON.parse(new TextDecoder().decode(fromBase64Url(str))));
}
