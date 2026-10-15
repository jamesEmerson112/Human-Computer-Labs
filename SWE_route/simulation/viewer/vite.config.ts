import { defineConfig, type ProxyOptions } from 'vite';

const backendOrigin = 'http://127.0.0.1:8082';
const devHosts = new Set(['127.0.0.1:5173', 'localhost:5173']);

/** Check the original browser origin before translating local proxy headers. */
function localProxy(websocket = false): ProxyOptions {
  return {
    target: backendOrigin,
    changeOrigin: true,
    ws: websocket,
    bypass(request) {
      const host = request.headers.host;
      const origin = request.headers.origin;
      if (!host || !devHosts.has(host)
        || (websocket && !origin)
        || (origin !== undefined && origin !== `http://${host}`)) return false;
      if (origin !== undefined) request.headers.origin = backendOrigin;
    },
  };
}

export default defineConfig({
  server: {
    host: '127.0.0.1',
    port: 5173,
    strictPort: true,
    proxy: {
      '/api': localProxy(),
      '/assets': localProxy(),
      '/ws': localProxy(true),
    },
  },
  build: { assetsDir: 'static' },
});
