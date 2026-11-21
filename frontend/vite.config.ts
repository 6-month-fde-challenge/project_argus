import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { fileURLToPath, URL } from "node:url";

// Dev server proxies /api to the platform (docker: http://localhost:8090); production is served by nginx.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: { alias: { "@": fileURLToPath(new URL("./src", import.meta.url)) } },
  server: { port: 3000, proxy: { "/api": { target: "http://localhost:8090", changeOrigin: true } } },
  build: {
    chunkSizeWarningLimit: 1500,
    rollupOptions: {
      output: {
        manualChunks: (id) =>
          id.includes("node_modules/echarts") || id.includes("node_modules/zrender") ? "charts"
          : id.includes("node_modules/@codemirror") || id.includes("node_modules/@uiw") ? "editor"
          : id.includes("node_modules/@xyflow") ? "flow"
          : undefined,
      },
    },
  },
  test: { environment: "jsdom", globals: true, setupFiles: ["./src/test/setup.ts"], css: false },
});
