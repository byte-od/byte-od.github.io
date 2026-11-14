/**
 * WebGPU Hardware Probe & Canonical Re-Exports for MoS + SCache v4.0
 *
 * All tensor execution runs via 15 custom WGSL 1.0 compute shaders in `mosWebgpuEngine.ts`
 * (`shaders/mos_scache.wgsl`) and `BrowserScarStoreHTTP206` (`scarstoreHttp206Browser.ts`).
 */

export { BrowserMoSWebGPUEngine, checkMosShardsCached } from './mosWebgpuEngine';
export { BrowserScarStoreHTTP206 } from './scarstoreHttp206Browser';

export interface WebGPUProbeResult {
  supported: boolean;
  adapterLabel: string;
  maxBufferSizeMiB: number;
}

export async function probeWebGPUHardware(): Promise<WebGPUProbeResult> {
  try {
    if (typeof navigator === 'undefined' || !('gpu' in navigator) || !(navigator as any).gpu) {
      return { supported: false, adapterLabel: 'WebGPU indisponível', maxBufferSizeMiB: 0 };
    }
    const adapter = await (navigator as any).gpu.requestAdapter({
      powerPreference: 'high-performance',
    });
    if (!adapter) {
      return { supported: false, adapterLabel: 'Sem adaptador GPU', maxBufferSizeMiB: 0 };
    }
    let label = 'WebGPU GPU';
    const info = adapter.info || (adapter.requestAdapterInfo ? await adapter.requestAdapterInfo() : null);
    if (info) {
      const parts = [info.vendor, info.architecture, info.description].filter(Boolean);
      if (parts.length > 0) label = parts.join(' · ');
    }
    const maxBuf = adapter.limits?.maxBufferSize
      ? Math.round(adapter.limits.maxBufferSize / (1024 * 1024))
      : 256;
    return { supported: true, adapterLabel: label, maxBufferSizeMiB: maxBuf };
  } catch {
    return { supported: false, adapterLabel: 'Fallback CPU S1', maxBufferSizeMiB: 0 };
  }
}
