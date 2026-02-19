"""End-to-end hierarchical pipeline orchestrator."""

from marketmap.workers.market_ingestion import ingest_markets
from marketmap.workers.event_projection_worker import run_event_projection_pipeline
from marketmap.workers.neighborhood_worker import ingest_event_tags_task
from marketmap.workers.celery_app import app


@app.task(bind=True, name="marketmap.workers.pipeline_orchestrator.run_hierarchical_pipeline", max_retries=1)
def run_hierarchical_pipeline(self) -> dict:  # type: ignore[type-arg]
    steps: list[dict] = []
    steps.append(ingest_markets.apply().get())  # type: ignore[attr-defined]
    steps.append(ingest_event_tags_task.apply().get())  # type: ignore[attr-defined]
    steps.append(run_event_projection_pipeline.apply(kwargs={"force_reembed": True}).get())  # type: ignore[attr-defined]
    return {"status": "success", "steps": steps}
