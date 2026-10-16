import { Route, Routes } from "react-router-dom";
import { AppShell } from "./components/AppShell";
import { DashboardPage } from "./pages/DashboardPage";
import { MemoriesPage } from "./pages/MemoriesPage";
import { MemoryDetailPage } from "./pages/MemoryDetailPage";
import { RetrievalPage } from "./pages/RetrievalPage";
import { ContextPage } from "./pages/ContextPage";
import { AgentPage } from "./pages/AgentPage";
import { TracesPage } from "./pages/TracesPage";
import { TraceDetailPage } from "./pages/TraceDetailPage";
import { EvaluationPage } from "./pages/EvaluationPage";
import { MetricsPage } from "./pages/MetricsPage";
import { NotFoundPage } from "./pages/NotFoundPage";

/** The eight primary areas of the control plane, in demo order. */
export function App() {
  return (
    <Routes>
      <Route element={<AppShell />}>
        <Route path="/" element={<DashboardPage />} />
        <Route path="/memories" element={<MemoriesPage />} />
        <Route path="/memories/:memoryId" element={<MemoryDetailPage />} />
        <Route path="/retrieval" element={<RetrievalPage />} />
        <Route path="/context" element={<ContextPage />} />
        <Route path="/agent" element={<AgentPage />} />
        <Route path="/traces" element={<TracesPage />} />
        <Route path="/traces/:traceId" element={<TraceDetailPage />} />
        <Route path="/evaluation" element={<EvaluationPage />} />
        <Route path="/metrics" element={<MetricsPage />} />
        <Route path="*" element={<NotFoundPage />} />
      </Route>
    </Routes>
  );
}
