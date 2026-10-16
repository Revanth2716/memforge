import { describe, expect, it } from "vitest";
import { screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { MemoriesPage } from "../src/pages/MemoriesPage";
import { MemoryDetailPage } from "../src/pages/MemoryDetailPage";
import { installFetch, renderAt, renderWithParams } from "./helpers";
import {
  conflicts,
  duplicates,
  history,
  IDS,
  intelligence,
  memoryPage,
  relationships,
} from "./fixtures";

const HEALTH = {
  path: "/health",
  body: { status: "ok", service: "memforge", version: "1.0.0", storage: "sqlite" },
};

const routes = () => [{ path: "/memories", body: memoryPage }, HEALTH];

const detailRoutes = (id = IDS.PROFILE) => [
  { path: "/memories", body: memoryPage },
  {
    path: new RegExp(`^/memories/${id}$`),
    body: memoryPage.items.find((item) => item.id === id) ?? memoryPage.items[0]!,
  },
  { path: new RegExp(`^/memories/${id}/history$`), body: history },
  { path: new RegExp(`^/memories/${id}/intelligence$`), body: intelligence },
  { path: new RegExp(`^/memories/${id}/duplicates$`), body: duplicates },
  { path: new RegExp(`^/memories/${id}/conflicts$`), body: conflicts },
  { path: new RegExp(`^/memories/${id}/relationships$`), body: relationships },
];

describe("memory explorer", () => {
  it("lists a real page of memories with the backend's own totals", async () => {
    const fetch = installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    expect(await screen.findByRole("link", { name: memoryPage.items[0]!.content })).toBeDefined();
    expect(screen.getByText(/6 memories in the backend for this filter/)).toBeDefined();
    expect(fetch.calls[0]!.path).toBe("/memories?limit=25&offset=0&status=ACTIVE");
  });

  it("sends status and memory type to the backend", async () => {
    const user = userEvent.setup();
    const fetch = installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });

    await user.selectOptions(screen.getByLabelText("Status"), "ARCHIVED");
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("status=ARCHIVED"));

    await user.selectOptions(screen.getByLabelText("Memory type"), "FACT");
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("memory_type=FACT"));
  });

  it("applies the confidence and importance thresholds to the loaded page and says so", async () => {
    const user = userEvent.setup();
    installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });

    await user.type(screen.getByLabelText("Min confidence"), "0.8");
    await waitFor(() => expect(screen.getByText(/after page-level filters/)).toBeDefined());
    // PROFILE (0.95) and EVENT (0.90) clear a 0.80 confidence floor; the header row
    // is the third row.
    expect(screen.getAllByRole("row")).toHaveLength(3);
  });

  it("filters the loaded page by content substring", async () => {
    const user = userEvent.setup();
    installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });
    await user.type(screen.getByLabelText("Filter loaded page"), "Kubernetes");
    await waitFor(() => expect(screen.queryByText(/User prefers Python/)).toBeNull());
    expect(screen.getByText(/The staging cluster runs on Kubernetes/)).toBeDefined();
  });

  it("sorts the loaded page by the chosen key", async () => {
    const user = userEvent.setup();
    installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });
    await user.selectOptions(screen.getByLabelText("Sort (page)"), "confidence");
    await waitFor(() => {
      const rows = screen.getAllByRole("row").slice(1);
      expect(rows[0]).toHaveTextContent("User prefers Python for backend work.");
    });
  });

  it("paginates with the backend's limit and offset", async () => {
    const user = userEvent.setup();
    const fetch = installFetch([
      { path: "/memories", body: { ...memoryPage, total: 60, limit: 25, offset: 0 } },
    ]);
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });

    await user.click(screen.getByRole("button", { name: /Next/ }));
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("offset=25"));
    expect(screen.getByText("60")).toBeDefined();
  });

  it("changes the page size the backend is asked for", async () => {
    const user = userEvent.setup();
    const fetch = installFetch(routes());
    renderAt(<MemoriesPage />, "/memories");
    await screen.findByRole("link", { name: memoryPage.items[0]!.content });
    await user.selectOptions(screen.getByLabelText("Page size"), "50");
    await waitFor(() => expect(fetch.calls.at(-1)!.path).toContain("limit=50"));
  });

  it("shows an empty state when the page has no records", async () => {
    installFetch([{ path: "/memories", body: { items: [], total: 0, limit: 25, offset: 0 } }]);
    renderAt(<MemoriesPage />, "/memories");
    expect(await screen.findByText("No memories match this page")).toBeDefined();
  });

  it("shows an API error with the backend's error code", async () => {
    installFetch([
      {
        path: "/memories",
        status: 503,
        body: { error: { code: "STORAGE_UNAVAILABLE", message: "SQLite storage did not answer" } },
      },
    ]);
    renderAt(<MemoriesPage />, "/memories");
    expect(await screen.findByText("STORAGE_UNAVAILABLE")).toBeDefined();
    expect(screen.getByText(/SQLite storage did not answer/)).toBeDefined();
  });

  it("shows a loading state while the page is in flight", async () => {
    installFetch([{ path: "/memories", pending: true }]);
    renderAt(<MemoriesPage />, "/memories");
    await waitFor(() => expect(document.body.textContent).toContain("Loading…"));
  });

  it("reports a malformed page without rendering invented rows", async () => {
    installFetch([{ path: "/memories", body: { items: "nope", total: 1, limit: 25, offset: 0 } }]);
    renderAt(<MemoriesPage />, "/memories");
    expect(await screen.findByText("Malformed API response")).toBeDefined();
    expect(screen.queryByRole("link", { name: memoryPage.items[0]!.content })).toBeNull();
  });
});

describe("memory detail", () => {
  it("renders every requested field of the memory", async () => {
    installFetch(detailRoutes());
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText(memoryPage.items[0]!.content)).toBeDefined();
    expect(screen.getByText("PROFILE")).toBeDefined();
    expect(screen.getByText("ACTIVE")).toBeDefined();
    expect(screen.getAllByText("0.950").length).toBeGreaterThan(0);
    expect(screen.getAllByText("0.900").length).toBeGreaterThan(0);
    expect(screen.getByText("user:demo")).toBeDefined();
  });

  it("renders the visual version history: v1 CREATED, v2 UPDATED, v3 FORGOTTEN", async () => {
    installFetch(detailRoutes());
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    // Wait for the history rows themselves, not just the card title.
    expect(await screen.findByText("FORGOTTEN")).toBeDefined();
    const versions = document.querySelectorAll(".version");
    expect(versions).toHaveLength(3);
    expect(versions[0]!.querySelector(".version__number")!.textContent).toBe("v1");
    expect(versions[0]!.querySelector(".version__change")!.textContent).toBe("CREATED");
    expect(versions[1]!.querySelector(".version__number")!.textContent).toBe("v2");
    expect(versions[1]!.querySelector(".version__change")!.textContent).toBe("UPDATED");
    expect(versions[2]!.querySelector(".version__number")!.textContent).toBe("v3");
    expect(versions[2]!.querySelector(".version__change")!.textContent).toBe("FORGOTTEN");
    // The newest version is marked current and carries the DELETED status.
    expect(versions[2]!.textContent).toContain("current");
    expect(versions[2]!.textContent).toContain("status DELETED");
    // The update is explained, not just shown.
    expect(versions[1]!.textContent).toContain("changed: importance, source (from v1)");
  });

  it("renders the computed intelligence: importance score, decay, effective priority", async () => {
    installFetch(detailRoutes());
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText("Importance score")).toBeDefined();
    expect(screen.getAllByText("0.810").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Effective priority").length).toBeGreaterThan(0);
    expect(screen.getAllByText("Decay multiplier").length).toBeGreaterThan(0);
    expect(screen.getByText("Access count")).toBeDefined();
    expect(screen.getByText(/2 near-duplicate active memories/)).toBeDefined();
    expect(screen.getByText(/caller supplied importance 0.90/)).toBeDefined();
  });

  it("renders near-duplicates with their similarity and shared tokens", async () => {
    installFetch(detailRoutes());
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText(/Dice=0.9091/)).toBeDefined();
    expect(screen.getByText(/Dice=0.6667/)).toBeDefined();
    expect(screen.getByText(/^shared: backend, prefer, python, user, work$/)).toBeDefined();
    expect(screen.getByText("Near-duplicates")).toBeDefined();
  });

  it("renders the conflict pair and resolves it on demand", async () => {
    const user = userEvent.setup();
    installFetch([
      ...detailRoutes(IDS.FACT_TS),
      {
        method: "POST",
        path: new RegExp(`/memories/${IDS.FACT_TS}/conflicts/${IDS.FACT_PYTHON}/resolve$`),
        body: {
          winner_memory_id: IDS.FACT_TS,
          loser_memory_id: IDS.FACT_PYTHON,
          decided_by: "importance",
          explanation: "the TypeScript memory has the higher importance",
          steps: [
            { factor: "importance", left_value: "0.90", right_value: "0.50", winner_id: IDS.FACT_TS },
          ],
          loser_archived: true,
          conflict: conflicts.items[0]!,
        },
      },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.FACT_TS}`, "/memories/:memoryId");
    const button = await screen.findByRole("button", { name: /Resolve deterministically/ });
    expect(button).toBeDefined();
    await user.click(button);
    await waitFor(() => expect(screen.getByText("resolved")).toBeDefined());
    expect(screen.getByText(/the TypeScript memory has the higher importance/)).toBeDefined();
    expect(screen.getByText(/The loser was archived/)).toBeDefined();
    expect(screen.getByText(/importance: 0.90 vs 0.50/)).toBeDefined();
  });

  it("renders the recorded relationships", async () => {
    installFetch(detailRoutes());
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText("Relationships")).toBeDefined();
    await waitFor(() => expect(screen.getAllByText("DUPLICATE_OF")).toHaveLength(2));
    expect(screen.getByText(/similarity 0.909/)).toBeDefined();
  });

  it("reports a 404 on a memory that does not exist", async () => {
    installFetch([
      {
        path: new RegExp(`^/memories/${IDS.PROFILE}$`),
        status: 404,
        body: { error: { code: "MEMORY_NOT_FOUND", message: "no memory with id 'x'" } },
      },
      { path: "/memories", body: memoryPage },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect((await screen.findAllByText("MEMORY_NOT_FOUND")).length).toBeGreaterThan(0);
    expect(screen.queryByText("Version history")).toBeNull();
  });

  it("still shows the history of a forgotten memory, because nothing is destroyed", async () => {
    installFetch([
      {
        path: new RegExp(`^/memories/${IDS.PROFILE}$`),
        status: 404,
        body: { error: { code: "MEMORY_NOT_FOUND", message: "no memory with id 'x'" } },
      },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/history$`), body: history },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");

    expect(await screen.findByText("This memory was forgotten")).toBeDefined();
    expect(
      screen.getByText(/Nothing was destroyed: the memory is soft-deleted/),
    ).toBeDefined();
    // The append-only versions are still readable.
    expect(await screen.findByText("FORGOTTEN")).toBeDefined();
    const versions = document.querySelectorAll(".version");
    expect(versions).toHaveLength(3);
  });

  it("shows an empty state for a memory with no relationship", async () => {
    installFetch([
      { path: new RegExp(`^/memories/${IDS.PROFILE}$`), body: memoryPage.items[0] },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/history$`), body: history },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/intelligence$`), body: intelligence },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/duplicates$`), body: duplicates },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/conflicts$`), body: conflicts },
      {
        path: new RegExp(`^/memories/${IDS.PROFILE}/relationships$`),
        body: { memory_id: IDS.PROFILE, total: 0, items: [] },
      },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText("No relationship recorded")).toBeDefined();
  });

  it("shows an empty state for a memory with no near-duplicate", async () => {
    installFetch([
      { path: new RegExp(`^/memories/${IDS.PROFILE}$`), body: memoryPage.items[0] },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/history$`), body: history },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/intelligence$`), body: intelligence },
      {
        path: new RegExp(`^/memories/${IDS.PROFILE}/duplicates$`),
        body: { memory_id: IDS.PROFILE, threshold: 0.6, total: 0, items: [] },
      },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/conflicts$`), body: conflicts },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/relationships$`), body: relationships },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText("No near-duplicate detected")).toBeDefined();
  });

  it("shows an empty state for a memory with no unresolved conflict", async () => {
    installFetch([
      { path: new RegExp(`^/memories/${IDS.PROFILE}$`), body: memoryPage.items[0] },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/history$`), body: history },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/intelligence$`), body: intelligence },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/duplicates$`), body: duplicates },
      {
        path: new RegExp(`^/memories/${IDS.PROFILE}/conflicts$`),
        body: { memory_id: IDS.PROFILE, total: 0, items: [] },
      },
      { path: new RegExp(`^/memories/${IDS.PROFILE}/relationships$`), body: relationships },
    ]);
    renderWithParams(<MemoryDetailPage />, `/memories/${IDS.PROFILE}`, "/memories/:memoryId");
    expect(await screen.findByText("No unresolved conflict")).toBeDefined();
  });
});
