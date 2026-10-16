/**
 * The application shell: sidebar navigation, a header that always names the
 * current page, and the single content area every route renders into.
 */

import { NavLink, Outlet, useLocation } from "react-router-dom";
import { useEffect, useMemo } from "react";
import { api } from "../api/client";
import { useResource } from "../hooks/useResource";
import { useTheme } from "../hooks/useTheme";
import { Badge } from "./primitives";

interface NavItem {
  to: string;
  label: string;
  hint: string;
}

const SECTIONS: readonly { title: string; items: readonly NavItem[] }[] = [
  {
    title: "Overview",
    items: [{ to: "/", label: "Dashboard", hint: "Live summary of the running system" }],
  },
  {
    title: "Memory",
    items: [
      { to: "/memories", label: "Memories", hint: "Explore the stored, versioned corpus" },
      { to: "/retrieval", label: "Retrieval Lab", hint: "Run a search and read the ranking" },
      { to: "/context", label: "Context Compiler", hint: "Compile a budgeted agent context" },
      { to: "/agent", label: "Agent Playground", hint: "One memory-aware agent turn" },
    ],
  },
  {
    title: "Observability",
    items: [
      { to: "/traces", label: "Traces", hint: "What a request actually did" },
      { to: "/evaluation", label: "Evaluation", hint: "The deterministic benchmark" },
      { to: "/metrics", label: "Metrics", hint: "Counters derived from recorded events" },
    ],
  },
];

const TITLES: readonly { prefix: string; title: string; eyebrow: string }[] = [
  { prefix: "/", title: "Dashboard", eyebrow: "Overview" },
  { prefix: "/memories/", title: "Memory detail", eyebrow: "M1-M2 · versioned record" },
  { prefix: "/memories", title: "Memories", eyebrow: "Memory core" },
  { prefix: "/retrieval", title: "Retrieval Lab", eyebrow: "M3 · memory-aware ranking" },
  { prefix: "/context", title: "Context Compiler", eyebrow: "M4 · token-budgeted context" },
  { prefix: "/agent", title: "Agent Playground", eyebrow: "M5 · bounded memory-aware agent" },
  { prefix: "/traces/", title: "Trace", eyebrow: "M6 · recorded lifecycle" },
  { prefix: "/traces", title: "Traces", eyebrow: "M6 · append-only audit trail" },
  { prefix: "/evaluation", title: "Evaluation", eyebrow: "M6 · deterministic benchmark" },
  { prefix: "/metrics", title: "Metrics", eyebrow: "M6 · operational counters" },
];

function usePageMeta(): { title: string; eyebrow: string } {
  const { pathname } = useLocation();
  const match = useMemo(
    () =>
      [...TITLES]
        .filter((entry) => entry.prefix !== "/" && pathname.startsWith(entry.prefix))
        .sort((left, right) => right.prefix.length - left.prefix.length)[0] ??
      TITLES[0]!,
    [pathname],
  );
  return { title: match.title, eyebrow: match.eyebrow };
}

function ConnectionBadge() {
  const { state } = useResource("health", () => api.health(), { ttlMs: 10_000 });
  const label =
    state.status === "ready"
      ? `API ${state.data.status} · v${state.data.version}`
      : state.status === "error"
        ? "API unreachable"
        : "checking API…";
  const tone = state.status === "ready" ? "ok" : state.status === "error" ? "error" : "loading";
  return (
    <span className="connection" data-state={tone} title={label}>
      <span className="connection__dot" aria-hidden="true" />
      <span>{label}</span>
    </span>
  );
}

export function AppShell() {
  const { title, eyebrow } = usePageMeta();
  const { theme, toggle } = useTheme();

  // Every route change should start at the top of the content area.
  const { pathname } = useLocation();
  useEffect(() => {
    window.scrollTo({ top: 0 });
  }, [pathname]);

  let index = 0;
  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand__mark" aria-hidden="true">
            MF
          </span>
          <span>
            <span className="brand__name">MemForge</span>
            <span className="brand__tag">control plane</span>
          </span>
        </div>

        <nav className="nav" aria-label="Dashboard sections">
          {SECTIONS.map((section) => (
            <div key={section.title}>
              <div className="nav__section">{section.title}</div>
              {section.items.map((item) => {
                index += 1;
                return (
                  <NavLink
                    key={item.to}
                    to={item.to}
                    end={item.to === "/"}
                    className="nav__link"
                    title={item.hint}
                  >
                    <span className="nav__index">{String(index).padStart(2, "0")}</span>
                    <span>{item.label}</span>
                  </NavLink>
                );
              })}
            </div>
          ))}
        </nav>

        <div className="sidebar__footer">
          <ConnectionBadge />
          <span>
            A presentation layer over the real MemForge APIs. No keys, no second
            backend.
          </span>
        </div>
      </aside>

      <div className="main">
        <header className="header">
          <div className="header__titles">
            <div className="header__eyebrow">{eyebrow}</div>
            <h1 className="header__title">{title}</h1>
          </div>
          <div className="header__actions">
            <Badge tone="accent">local-first · offline by default</Badge>
            <button
              type="button"
              className="btn btn--ghost"
              onClick={toggle}
              aria-label={`Switch to ${theme === "dark" ? "light" : "dark"} theme`}
            >
              {theme === "dark" ? "☾ Dark" : "☀ Light"}
            </button>
          </div>
        </header>
        <main className="content">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
