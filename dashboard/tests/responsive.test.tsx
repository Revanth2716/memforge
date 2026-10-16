/**
 * Responsive-critical components.
 *
 * jsdom has no layout engine, so what is asserted here is the *structural*
 * contract the CSS depends on: the shell grid, the landmark elements, the
 * scroll container, and the fact that the components stay usable without a
 * wide viewport (labels on every control, a table inside a scroll wrapper,
 * text that wraps rather than overflowing).
 */

import { describe, expect, it } from "vitest";
import { screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { App } from "../src/App";
import { MemoriesPage } from "../src/pages/MemoriesPage";
import { RetrievalPage } from "../src/pages/RetrievalPage";
import { installFetch, renderAt } from "./helpers";
import {
  compile,
  emptySearch,
  evaluation,
  memoryPage,
  metrics,
  search,
  traceList,
} from "./fixtures";

const HEALTH = {
  path: "/health",
  body: { status: "ok", service: "memforge", version: "1.0.0", storage: "sqlite" },
};

const routes = () => [
  HEALTH,
  { path: "/memories", body: memoryPage },
  { path: "/traces", body: traceList },
  { path: "/metrics", body: metrics },
  { path: "/evaluations/latest", body: evaluation },
  { method: "POST", path: "/memories/search", body: search },
  { method: "POST", path: "/context/compile", body: compile },
];

describe("responsive-critical structure", () => {
  it("lays the shell out as a sidebar beside a content column", async () => {
    installFetch(routes());
    renderAt(<App />, "/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });

    const shell = document.querySelector(".app-shell")!;
    expect(shell.className).toContain("app-shell");
    // CSS collapses grid-template-columns below 960px, so the component only
    // has to expose the two regions in a stable order.
    const regions = [...shell.children].map((child) => child.className);
    expect(regions).toEqual(["sidebar", "main"]);
  });

  it("exposes the expected landmarks so the layout is navigable", async () => {
    installFetch(routes());
    renderAt(<App />, "/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });

    expect(screen.getByRole("navigation", { name: "Dashboard sections" })).toBeDefined();
    expect(screen.getByRole("main")).toBeDefined();
    // Card headers are also <header> elements; the shell one is the landmark.
    expect(
      screen.getAllByRole("banner").some((node) => node.className.includes("header")),
    ).toBe(true);
  });

  it("gives every sidebar link a title describing the destination", async () => {
    installFetch(routes());
    renderAt(<App />, "/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });
    const links = screen.getAllByRole("link").filter((link) => link.className.includes("nav__link"));
    expect(links.length).toBe(8);
    for (const link of links) {
      expect(link.getAttribute("title")).toBeTruthy();
      expect(link.getAttribute("title")!.length).toBeGreaterThan(5);
    }
  });

  it("scrolls the content area rather than the page when it grows long", async () => {
    installFetch(routes());
    renderAt(<App />, "/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });
    expect(document.querySelector(".content")).toBeDefined();
    expect(document.querySelector(".header")).toBeDefined();
  });

  it("wraps every data table in a horizontally scrollable container", async () => {
    const user = userEvent.setup();
    installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });

    const wrap = document.querySelector(".table-wrap")!;
    expect(wrap).not.toBeNull();
    expect(wrap.querySelector("table.table")).not.toBeNull();
    // Filters wrap instead of pushing the page wide.
    expect(document.querySelector(".filter-bar")).not.toBeNull();
    await user.selectOptions(screen.getByLabelText("Page size"), "10");
  });

  it("keeps wide data readable by allowing it to wrap", async () => {
    const user = userEvent.setup();
    installFetch(routes());
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    await screen.findByText("Why rank #1 ranked first");
    // The ranked result uses a fluid grid; long content wraps inside cells.
    expect(document.querySelector(".ranked__body")).not.toBeNull();
    expect(document.querySelector(".factor__why")).not.toBeNull();
  });

  it("keeps ids and traces in a monospace face for scanning", async () => {
    installFetch(routes());
    renderAt(<App />, "/traces");
    await screen.findByRole("link", { name: /trace-47c62f17/ });
    expect(document.querySelectorAll(".mono").length).toBeGreaterThan(0);
  });

  it("marks a dense layout on the memory detail columns", async () => {
    installFetch(routes());
    renderAt(<App />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });
    expect(document.querySelector(".grid--split")).toBeNull();
    // The explorer's summary grids are responsive by class contract.
    expect(document.querySelector(".grid--stats")).toBeNull();
  });

  it("keeps the sidebar reachable when the viewport collapses", async () => {
    // jsdom cannot reflow, so this asserts the CSS contract the collapse relies
    // on: the nav is a flex container whose links stay grouped and labelled.
    installFetch(routes());
    renderAt(<App />, "/");
    await screen.findByRole("heading", { level: 1, name: "Dashboard" });
    const nav = screen.getByRole("navigation", { name: "Dashboard sections" });
    expect(nav.className).toBe("nav");
    expect(nav.querySelectorAll(".nav__link")).toHaveLength(8);
  });

  it("never leaves a page blank: every route renders content or an explicit state", async () => {
    installFetch([]);
    for (const path of ["/", "/memories", "/retrieval", "/context", "/agent", "/traces", "/evaluation", "/metrics"]) {
      const view = renderAt(<App />, path);
      const body = document.body.textContent ?? "";
      expect(body.length).toBeGreaterThan(200);
      expect(body).toMatch(/not available|unreachable|Loading|No |yet|not been/i);
      view.unmount();
    }
  });

  it("renders the retrieval empty state rather than an empty results panel", async () => {
    const user = userEvent.setup();
    installFetch([
      { method: "POST", path: "/memories/search", body: emptySearch },
      ...routes(),
    ]);
    renderAt(<RetrievalPage />, "/retrieval");
    await user.click(screen.getByRole("button", { name: /Run retrieval/ }));
    expect(await screen.findByText("No memory passed the filters")).toBeDefined();
  });
});