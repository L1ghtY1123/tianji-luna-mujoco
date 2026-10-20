import { defineConfig } from 'vite';

const backendUrl = process.env.JEV_DEV_BACKEND_URL || 'http://127.0.0.1:8340';
const websocketUrl = backendUrl.replace(/^http/, 'ws');

export default defineConfig({
  build: {
    assetsDir: 'web-assets',
    rollupOptions: { input: ['index.html', 'collision-spheres.html', 'replay.html', 'workspaces.html'] },
  },
  server: {
    proxy: {
      '/api': backendUrl,
      '/assets': backendUrl,
      '/ws': { target: websocketUrl, ws: true },
    },
  },
});
