"""Event-first projection worker pipeline."""

from datetime import datetime, timezone

from marketmap.events.pipeline import (
    assign_event_neighborhoods,
    compute_event_embeddings,
    project_events_hierarchical,
)
from marketmap.models.database import SyncSessionLocal
from marketmap.workers.celery_app import app


@app.task(bind=True, name="marketmap.workers.event_projection_worker.run_event_projection_pipeline", max_retries=2)
def run_event_projection_pipeline(self, force_reembed: bool = True) -> dict:  # type: ignore[type-arg]
    started = datetime.now(timezone.utc)
    session = SyncSessionLocal()
    try:
        a = assign_event_neighborhoods(session)
        b = compute_event_embeddings(session, force_reembed=force_reembed)
        c = project_events_hierarchical(session)
        return {
            "status": "success",
            "assign": a,
            "embed": b,
            "project": c,
            "elapsed_seconds": (datetime.now(timezone.utc) - started).total_seconds(),
        }
    except Exception as exc:
        session.rollback()
        raise self.retry(exc=exc, countdown=120)
    finally:
        session.close()
