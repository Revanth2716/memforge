/// <reference types="vite/client" />
/// <reference types="vitest" />
import "@testing-library/jest-dom/vitest";

// jsdom defines window.scrollTo but raises "not implemented"; the shell calls
// it on every route change, so tests need a working no-op.
window.scrollTo = (): void => {
  /* no-op in tests */
};
