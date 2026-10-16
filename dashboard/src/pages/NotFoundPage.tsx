import { Link } from "react-router-dom";
import { Card } from "../components/primitives";

/** A route that is not one of the dashboard's pages. */
export function NotFoundPage() {
  return (
    <>
      <Card title="No such page" subtitle="The control plane has eight areas.">
        <div className="stack">
          <p className="muted">
            That address is not part of the dashboard. The API itself is documented at{" "}
            <code>/docs</code> and described at <code>/openapi.json</code>.
          </p>
          <div className="btn-row">
            <Link className="btn btn--primary" to="/">
              Go to the dashboard
            </Link>
            <Link className="btn" to="/retrieval">
              Retrieval Lab
            </Link>
            <Link className="btn" to="/evaluation">
              Evaluation
            </Link>
          </div>
        </div>
      </Card>
    </>
  );
}
