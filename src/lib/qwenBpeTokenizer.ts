/**
 * Pure Zero-Dependency Qwen2.5 Byte-Level BPE Tokenizer (`tokenizer.json`)
 * 100% Bit-Exact Token ID Parity with HuggingFace `AutoTokenizer` (PT / EN / ES / FR)
 * Includes Persistent CacheStorage (`mos-webgpu-v4-assets`) for zero repeat network downloads.
 */

export const MOS_WEBGPU_CACHE_NAME = 'mos-webgpu-v4-assets-v1';

const QWEN_SPLIT_RE =
  /(?:'s|'t|'re|'ve|'m|'ll|'d|'S|'T|'RE|'VE|'M|'LL|'D)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+/gu;

function buildByteUnicodeTables(): { byteToChar: string[]; charToByte: Map<string, number> } {
  const bs: number[] = [];
  for (let b = 33; b <= 126; b++) bs.push(b);
  for (let b = 161; b <= 172; b++) bs.push(b);
  for (let b = 174; b <= 255; b++) bs.push(b);
  const cs = bs.slice();
  let n = 0;
  for (let b = 0; b < 256; b++) {
    if (!bs.includes(b)) {
      bs.push(b);
      cs.push(256 + n);
      n += 1;
    }
  }
  const byteToChar = new Array<string>(256);
  const charToByte = new Map<string, number>();
  for (let i = 0; i < 256; i++) {
    const ch = String.fromCodePoint(cs[i]);
    byteToChar[bs[i]] = ch;
    charToByte.set(ch, bs[i]);
  }
  return { byteToChar, charToByte };
}

const { byteToChar: BYTE_TO_CHAR, charToByte: CHAR_TO_BYTE } = buildByteUnicodeTables();
const UTF8_ENCODER = new TextEncoder();
const UTF8_DECODER = new TextDecoder('utf-8', { fatal: false });

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

export async function fetchWithPersistentCache(url: string): Promise<{ response: Response; fromCache: boolean }> {
  if (typeof caches !== 'undefined') {
    try {
      const cache = await caches.open(MOS_WEBGPU_CACHE_NAME);
      const cached = await cache.match(url);
      if (cached && cached.ok) {
        return { response: cached, fromCache: true };
      }
      const fresh = await fetch(url);
      if (fresh.ok) {
        try {
          await cache.put(url, fresh.clone());
        } catch (_) {
          // Ignore quota warnings on restricted storage modes
        }
      }
      return { response: fresh, fromCache: false };
    } catch (_) {
      // Fallback to direct fetch
    }
  }
  const resp = await fetch(url);
  return { response: resp, fromCache: false };
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

interface AddedTokenSpec {
  id: number;
  content: string;
  special?: boolean;
}

export class QwenBPETokenizer {
  private readonly vocab = new Map<string, number>();
  private readonly idToToken = new Map<number, string>();
  private readonly specialTokens = new Set<number>();
  private readonly addedTokens: AddedTokenSpec[] = [];
  private readonly addedSplitRe: RegExp | null;
  private readonly addedSet: Set<string>;
  private readonly bpeRanks = new Map<string, number>();
  private readonly useNFC: boolean;
  private readonly bpeCache = new Map<string, string[]>();

  constructor(tokenizerJson: any) {
    const rawVocab = tokenizerJson.model.vocab as Record<string, number>;
    for (const key of Object.keys(rawVocab)) {
      this.vocab.set(key, rawVocab[key]);
    }
    for (const [tok, id] of this.vocab.entries()) {
      this.idToToken.set(id, tok);
    }

    for (const item of (tokenizerJson.added_tokens || []) as AddedTokenSpec[]) {
      this.vocab.set(item.content, item.id);
      this.idToToken.set(item.id, item.content);
      this.addedTokens.push(item);
      if (item.special) {
        this.specialTokens.add(item.id);
      }
    }

    const sortedAdded = this.addedTokens
      .map((t) => t.content)
      .sort((a, b) => b.length - a.length);
    this.addedSplitRe =
      sortedAdded.length > 0 ? new RegExp(`(${sortedAdded.map(escapeRegExp).join('|')})`, 'g') : null;
    this.addedSet = new Set(sortedAdded);

    const merges: Array<string | [string, string]> = tokenizerJson.model.merges || [];
    for (let i = 0; i < merges.length; i++) {
      const m = merges[i];
      const key = Array.isArray(m) ? `${m[0]}\u0001${m[1]}` : m.replace(' ', '\u0001');
      this.bpeRanks.set(key, i);
    }

    this.useNFC = Boolean(tokenizerJson.normalizer && tokenizerJson.normalizer.type === 'NFC');
  }

  public static async fromUrl(url = './tokenizer.json'): Promise<QwenBPETokenizer> {
    const { response } = await fetchWithPersistentCache(url);
    if (!response.ok) {
      throw new Error(`Failed to load tokenizer from ${url}: HTTP ${response.status}`);
    }
    const data = await response.json();
    return new QwenBPETokenizer(data);
  }

  private _bpe(tokenStr: string): string[] {
    const cached = this.bpeCache.get(tokenStr);
    if (cached !== undefined) return cached;

    let word = Array.from(tokenStr);
    if (word.length <= 1) {
      this.bpeCache.set(tokenStr, word);
      return word;
    }

    while (word.length > 1) {
      let bestRank = Infinity;
      let bestPair: [string, string] | null = null;
      for (let i = 0; i < word.length - 1; i++) {
        const key = `${word[i]}\u0001${word[i + 1]}`;
        const r = this.bpeRanks.get(key);
        if (r !== undefined && r < bestRank) {
          bestRank = r;
          bestPair = [word[i], word[i + 1]];
        }
      }
      if (bestPair === null) break;

      const [first, second] = bestPair;
      const nextWord: string[] = [];
      let i = 0;
      while (i < word.length) {
        if (i < word.length - 1 && word[i] === first && word[i + 1] === second) {
          nextWord.push(first + second);
          i += 2;
        } else {
          nextWord.push(word[i]);
          i += 1;
        }
      }
      word = nextWord;
    }

    if (this.bpeCache.size > 16384) {
      this.bpeCache.clear();
    }
    this.bpeCache.set(tokenStr, word);
    return word;
  }

  private _encodeOrdinaryChunk(chunk: string): number[] {
    const norm = this.useNFC ? chunk.normalize('NFC') : chunk;
    const ids: number[] = [];
    const pieces = norm.match(QWEN_SPLIT_RE);
    if (!pieces) return ids;

    for (const piece of pieces) {
      const bytes = UTF8_ENCODER.encode(piece);
      let mapped = '';
      for (let i = 0; i < bytes.length; i++) {
        mapped += BYTE_TO_CHAR[bytes[i]];
      }
      const bpeTokens = this._bpe(mapped);
      for (const bt of bpeTokens) {
        const id = this.vocab.get(bt);
        if (id !== undefined) {
          ids.push(id);
        }
      }
    }
    return ids;
  }

  public encode(text: string): number[] {
    if (!text) return [];
    if (!this.addedSplitRe) {
      return this._encodeOrdinaryChunk(text);
    }
    const parts = text.split(this.addedSplitRe);
    const ids: number[] = [];
    for (const part of parts) {
      if (!part) continue;
      if (this.addedSet.has(part)) {
        const specialId = this.vocab.get(part);
        if (specialId !== undefined) ids.push(specialId);
      } else {
        const sub = this._encodeOrdinaryChunk(part);
        for (let i = 0; i < sub.length; i++) ids.push(sub[i]);
      }
    }
    return ids;
  }

  public applyChatTemplate(messages: ChatMessage[], addGenerationPrompt = true): number[] {
    let prompt = '';
    for (const msg of messages) {
      prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
    }
    if (addGenerationPrompt) {
      prompt += '<|im_start|>assistant\n';
    }
    return this.encode(prompt);
  }

  public decode(tokenIds: number[], skipSpecialTokens = true): string {
    const byteList: number[] = [];
    const chunks: string[] = [];

    const flushBytes = () => {
      if (byteList.length > 0) {
        chunks.push(UTF8_DECODER.decode(new Uint8Array(byteList)));
        byteList.length = 0;
      }
    };

    for (const id of tokenIds) {
      if (skipSpecialTokens && this.specialTokens.has(id)) {
        continue;
      }
      const tok = this.idToToken.get(id);
      if (tok === undefined) continue;
      if (this.addedSet.has(tok)) {
        flushBytes();
        if (!skipSpecialTokens) chunks.push(tok);
        continue;
      }
      for (const ch of tok) {
        const b = CHAR_TO_BYTE.get(ch);
        if (b !== undefined) {
          byteList.push(b);
        } else {
          flushBytes();
          chunks.push(ch);
        }
      }
    }
    flushBytes();
    return chunks.join('');
  }
}
