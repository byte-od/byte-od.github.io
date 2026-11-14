import tailwindcss from '@tailwindcss/vite';
import react from '@vitejs/plugin-react';
import fs from 'fs';
import path from 'path';
import type {Plugin} from 'vite';
import {defineConfig} from 'vite';

/**
 * RFC 7233 `HTTP/1.1 206 Partial Content` middleware for `.scarstore` files in `vite dev` & `vite preview`.
 * Mirrors GitHub Pages Fastly CDN (`byte-od.github.io`), which natively serves `HTTP/2 206` byte ranges.
 */
function scarstoreHttp206Plugin(): Plugin {
  const attachMiddleware = (middlewares: any, rootDirs: string[]) => {
    middlewares.use((req: any, res: any, next: any) => {
      const rawUrl: string = (req.url || '').split('?')[0];
      if (!rawUrl.endsWith('.scarstore')) {
        return next();
      }
      const relPath = decodeURIComponent(rawUrl).replace(/^\/+/, '');
      let filePath: string | null = null;
      for (const dir of rootDirs) {
        const candidate = path.resolve(dir, relPath);
        if (candidate.startsWith(path.resolve(dir)) && fs.existsSync(candidate) && fs.statSync(candidate).isFile()) {
          filePath = candidate;
          break;
        }
      }
      if (!filePath) {
        return next();
      }

      const stat = fs.statSync(filePath);
      const total = stat.size;
      const rangeHeader: string | undefined = req.headers?.range;

      res.setHeader('Accept-Ranges', 'bytes');
      res.setHeader('Content-Type', 'application/octet-stream');
      res.setHeader('Cache-Control', 'no-store');
      res.setHeader('Access-Control-Allow-Origin', '*');
      res.setHeader('Access-Control-Expose-Headers', 'Content-Range, Content-Length, Accept-Ranges');

      if (rangeHeader) {
        const m = /^bytes=(\d+)-(\d*)$/.exec(rangeHeader.trim());
        if (!m) {
          res.statusCode = 416;
          res.setHeader('Content-Range', `bytes */${total}`);
          res.end();
          return;
        }
        const start = parseInt(m[1], 10);
        const end = m[2] ? Math.min(parseInt(m[2], 10), total - 1) : total - 1;
        if (Number.isNaN(start) || Number.isNaN(end) || start > end || start >= total) {
          res.statusCode = 416;
          res.setHeader('Content-Range', `bytes */${total}`);
          res.end();
          return;
        }
        const chunkLen = end - start + 1;
        res.statusCode = 206;
        res.setHeader('Content-Range', `bytes ${start}-${end}/${total}`);
        res.setHeader('Content-Length', String(chunkLen));
        if (req.method === 'HEAD') {
          res.end();
          return;
        }
        fs.createReadStream(filePath, {start, end}).pipe(res);
        return;
      }

      res.statusCode = 200;
      res.setHeader('Content-Length', String(total));
      if (req.method === 'HEAD') {
        res.end();
        return;
      }
      fs.createReadStream(filePath).pipe(res);
    });
  };

  const root = import.meta.dirname;
  return {
    name: 'scarstore-http206-range',
    configureServer(server) {
      attachMiddleware(server.middlewares, [path.resolve(root, 'public'), root]);
    },
    configurePreviewServer(server) {
      attachMiddleware(server.middlewares, [path.resolve(root, 'dist'), path.resolve(root, 'public')]);
    },
  };
}

export default defineConfig(() => {
  return {
    base: process.env.BASE_URL || '/',
    plugins: [scarstoreHttp206Plugin(), react(), tailwindcss()],
    resolve: {
      alias: {
        '@': path.resolve(import.meta.dirname, '.'),
      },
    },
    build: {
      sourcemap: false,
    },
    server: {
      // HMR is disabled in AI Studio via DISABLE_HMR env var.
      // Do not modify—file watching is disabled to prevent flickering during agent edits.
      hmr: process.env.DISABLE_HMR !== 'true',
      // Disable file watching when DISABLE_HMR is true to save CPU during agent edits.
      watch: process.env.DISABLE_HMR === 'true' ? null : {},
    },
  };
});
