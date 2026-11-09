/**
 * WebGPU Hardware Dispatch Layer for MoS and SCache
 * Compiles WGSL shaders and manages compute pipelines on client GPU hardware.
 */

import wgslSource from './shaders/mos_scache.wgsl?raw';

export interface WebGPUExecutionStatus {
  supported: boolean;
  initialized: boolean;
  adapterName: string;
  backend: 'WebGPU (Hardware Accelerated)' | 'WASM/SIMD Float32Array (Fallback)';
}

export class WebGPUMoSRunner {
  private device: GPUDevice | null = null;
  private shaderModule: GPUShaderModule | null = null;
  private adapterName: string = 'Unknown';
  private isReady: boolean = false;

  public async initialize(): Promise<WebGPUExecutionStatus> {
    try {
      if (typeof navigator === 'undefined' || !('gpu' in navigator)) {
        return {
          supported: false,
          initialized: false,
          adapterName: 'None',
          backend: 'WASM/SIMD Float32Array (Fallback)',
        };
      }

      const adapter = await (navigator as any).gpu.requestAdapter({
        powerPreference: 'high-performance',
      });

      if (!adapter) {
        return {
          supported: false,
          initialized: false,
          adapterName: 'No suitable GPU adapter',
          backend: 'WASM/SIMD Float32Array (Fallback)',
        };
      }

      this.adapterName = adapter.info?.description || adapter.name || 'WebGPU Compatible GPU';
      this.device = await adapter.requestDevice();

      if (this.device && wgslSource) {
        this.shaderModule = this.device.createShaderModule({
          code: wgslSource,
        });
        this.isReady = true;
      }

      return {
        supported: true,
        initialized: this.isReady,
        adapterName: this.adapterName,
        backend: 'WebGPU (Hardware Accelerated)',
      };
    } catch (err) {
      console.warn('[WebGPUMoSRunner] WebGPU init notice (using CPU typed array fallback):', err);
      return {
        supported: false,
        initialized: false,
        adapterName: 'Fallback Device',
        backend: 'WASM/SIMD Float32Array (Fallback)',
      };
    }
  }

  public get isHardwareAccelerated(): boolean {
    return this.isReady && this.device !== null;
  }

  public get status(): WebGPUExecutionStatus {
    return {
      supported: typeof navigator !== 'undefined' && 'gpu' in navigator,
      initialized: this.isReady,
      adapterName: this.adapterName,
      backend: this.isReady
        ? 'WebGPU (Hardware Accelerated)'
        : 'WASM/SIMD Float32Array (Fallback)',
    };
  }
}

export const webgpuBridge = new WebGPUMoSRunner();
