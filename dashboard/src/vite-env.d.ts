/** Vite env bindings the dashboard reads. Only a URL is ever configurable. */
interface ImportMetaEnv {
  readonly VITE_MEMFORGE_API_BASE?: string;
}

interface ImportMeta {
  readonly env: ImportMetaEnv;
}
