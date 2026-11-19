import { registerHooks } from 'node:module';
import * as fs from 'node:fs';
import { fileURLToPath } from 'node:url';

registerHooks({
  load(url, context, nextLoad) {
    const cleanUrl = url.split('?')[0];
    if (cleanUrl.endsWith('.wgsl')) {
      const src = fs.readFileSync(fileURLToPath(cleanUrl), 'utf-8');
      return {
        format: 'module',
        shortCircuit: true,
        source: `export default ${JSON.stringify(src)};\n`,
      };
    }
    return nextLoad(url, context);
  },
});
