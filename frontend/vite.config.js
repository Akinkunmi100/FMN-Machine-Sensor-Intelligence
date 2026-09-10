import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// In dev the SPA runs on :5173 and proxies /api to the FastAPI process on
// :8000. In production `vite build` emits frontend/dist, which FastAPI serves
// from its own origin — so there is one Render service and no CORS at all.
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/api": {
        target: "http://127.0.0.1:8000",
        changeOrigin: true,
      },
    },
  },
  build: { outDir: "dist", emptyOutDir: true },
});
