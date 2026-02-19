"""Neighborhood ingestion and assignment workers."""

from marketmap.ingest.polymarket_event_tags import ingest_event_tags
from marketmap.neighborhoods.assign import assign_market_neighborhoods
from marketmap.workers.celery_app import app


@app.task(bind=True, name="marketmap.workers.neighborhood_worker.ingest_event_tags", max_retries=2)
def ingest_event_tags_task(self, path: str | None = None) -> dict:  # type: ignore[type-arg]
    return ingest_event_tags(path=path)


@app.task(bind=True, name="marketmap.workers.neighborhood_worker.assign_neighborhoods", max_retries=2)
def assign_neighborhoods_task(self) -> dict:  # type: ignore[type-arg]
    return assign_market_neighborhoods()
