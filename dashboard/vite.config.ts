/// <reference types="vitest" />
import react from "@vitejs/plugin-react";
import { defineConfig } from "vitest/config";

/**
 * Where the MemForge API lives.
 *
 * The dashboard talks to the **existing** FastAPI backend and holds no business
 * logic of its own. The default is a same-origin relative path, so the
 * production build can be served from anywhere as long as `/api` and `/health`
 * reach MemForge.
 *
 * Override for a backend on another origin, e.g.
 *   VITE_MEMFORGE_API_TARGET=http://127.0.0.1:8000   (dev/preview proxy target)
 *   VITE_MEMFORGE_API_BASE=http://127.0.0.1:8000     (baked into the bundle)
 *
 * These are URLs, never credentials: no API key, token or Authorization header
 * is configured anywhere in the frontend (see README, "Dashboard security").
 */
const apiTarget = process.env.VITE_MEMFORGE_API_TARGET ?? "http://127.0.0.1:8000";

/** Paths the backend owns. Everything else is the SPA. */
const proxyPaths = ["/api", "/health"] as const;

const proxy = Object.fromEntries(
  proxyPaths.map((path) => [path, { target: apiTarget, changeOrigin: true }]),
);

export default defineConfig({
  plugins: [react()],
  server: { port: 5173, proxy },
  preview: { port: 4173, proxy },
  build: {
    outDir: "dist",
    sourcemap: false,
    // The dashboard is one small bundle; splitting it would only add requests.
    chunkSizeWarningLimit: 700,
  },
  test: {
    environment: "jsdom",
    globals: true,
    setupFiles: ["./vitest.setup.ts"],
    include: ["tests/**/*.test.{ts,tsx}"],
    css: true,
    restoreMocks: true,
  },
});
