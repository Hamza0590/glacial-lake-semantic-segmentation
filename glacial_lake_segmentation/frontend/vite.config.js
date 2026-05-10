import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/predict':  'http://127.0.0.1:8000',
      '/evaluate': 'http://127.0.0.1:8000',
      '/health':   'http://127.0.0.1:8000',
      '/results':  'http://127.0.0.1:8000',
    }
  },
  build: {
    outDir: '../api/static',
    emptyOutDir: true,
  }
})
