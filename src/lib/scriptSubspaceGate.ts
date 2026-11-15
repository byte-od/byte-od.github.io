/**
 * Script-Orthogonal Active Subspace Gate (`P_⊥CJK` — WGSL Shader #18 Companion)
 *
 * Bit-for-bit TypeScript mirror of `scar_cache/vq4_scar_mos.py::has_cjk_script` and
 * `build_cjk_vocab_bitmask_u32` (`18.55 KB` = `4,748 uint32` words for `vocab_size = 151,936`).
 */

import type { QwenBPETokenizer } from './qwenBpeTokenizer';

/**
 * Exact Unicode codepoint check matching `scar_cache/vq4_scar_mos.py::has_cjk_script` bit-for-bit.
 */
export function hasCjkScript(text: string): boolean {
  for (const ch of text) {
    const cp = ch.codePointAt(0) ?? 0;
    if (
      (cp >= 0x3040 && cp <= 0x30ff) || // Hiragana + Katakana
      (cp >= 0x3400 && cp <= 0x4dbf) || // CJK Extension A
      (cp >= 0x4e00 && cp <= 0x9fff) || // CJK Unified Ideographs
      (cp >= 0xac00 && cp <= 0xd7af) || // Hangul Syllables
      (cp >= 0xf900 && cp <= 0xfaff) || // CJK Compatibility Ideographs
      (cp >= 0x3000 && cp <= 0x303f) || // CJK Symbols & Punctuation
      (cp >= 0xff01 && cp <= 0xff60) // Fullwidth CJK Forms
    ) {
      return true;
    }
  }
  return false;
}

/**
 * Builds the exact `ceil(vocabSize / 32)` `uint32` bitmask (`18.55 KB` for Qwen2.5) of CJK token IDs,
 * matching `scar_cache/vq4_scar_mos.py::build_cjk_vocab_bitmask_u32` bit-for-bit.
 */
export function buildCjkVocabBitmaskU32(tokenizer: QwenBPETokenizer, vocabSize: number): Uint32Array {
  const nWords = Math.ceil(vocabSize / 32);
  const maskU32 = new Uint32Array(nWords);
  for (let tid = 0; tid < vocabSize; tid++) {
    const piece = tokenizer.decode([tid], false);
    if (piece && hasCjkScript(piece)) {
      maskU32[tid >>> 5] = (maskU32[tid >>> 5] | (1 << (tid & 31))) >>> 0;
    }
  }
  return maskU32;
}
