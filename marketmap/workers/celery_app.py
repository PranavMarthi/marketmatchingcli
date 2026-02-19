"""Celery application configuration with Beat schedule."""

from celery import Celery
from celery.schedules import crontab

from marketmap.config import settings

app = Celery(
    "marketmap",
    broker=settings.celery_broker_url,
    backend=settings.redis_url,
    include=[
        "marketmap.workers.market_ingestion",
        "marketmap.workers.price_snapshot",
        "marketmap.workers.embedding_worker",
        "marketmap.workers.discovery_worker",
        "marketmap.workers.entity_worker",
        "marketmap.workers.hedge_worker",
        "marketmap.workers.projection_worker",
        "marketmap.workers.projection_distortion_worker",
        "marketmap.workers.memgraph_sync_worker",
        "marketmap.workers.clustering_worker",
        "marketmap.workers.neighborhood_worker",
        "marketmap.workers.local_clustering_worker",
        "marketmap.workers.pipeline_orchestrator",
        "marketmap.workers.event_projection_worker",
    ],
)

app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    # Retry settings
    task_default_retry_delay=60,
    task_default_max_retries=3,
)

# Celery Beat schedule
app.conf.beat_schedule = {
    # Milestone 1: Foundation workers
    "ingest-markets-every-5-min": {
        "task": "marketmap.workers.market_ingestion.ingest_markets",
        "schedule": settings.market_ingestion_interval_seconds,
        "options": {"queue": "default"},
    },
    "snapshot-prices-every-5-min": {
        "task": "marketmap.workers.price_snapshot.snapshot_prices",
        "schedule": settings.price_snapshot_interval_seconds,
        "options": {"queue": "default"},
    },
    # Milestone 2: Discovery graph workers (daily)
    "compute-embeddings-daily": {
        "task": "marketmap.workers.embedding_worker.compute_market_embeddings",
        "schedule": crontab(hour=2, minute=0),  # 2:00 AM UTC daily
        "options": {"queue": "default"},
    },
    "extract-entities-daily": {
        "task": "marketmap.workers.entity_worker.extract_market_entities",
        "schedule": crontab(hour=2, minute=30),  # 2:30 AM UTC daily
        "options": {"queue": "default"},
    },
    "compute-discovery-edges-daily": {
        "task": "marketmap.workers.discovery_worker.compute_discovery_edges",
        "schedule": crontab(hour=3, minute=0),  # 3:00 AM UTC daily (after embeddings)
        "options": {"queue": "default"},
    },
    "ingest-event-tags-daily": {
        "task": "marketmap.workers.neighborhood_worker.ingest_event_tags",
        "schedule": crontab(hour=3, minute=10),
        "options": {"queue": "default"},
    },
    "assign-neighborhoods-daily": {
        "task": "marketmap.workers.neighborhood_worker.assign_neighborhoods",
        "schedule": crontab(hour=3, minute=20),
        "options": {"queue": "default"},
    },
    # Milestone 3: Hedge graph workers (daily, after fresh price snapshots)
    "compute-hedge-edges-daily": {
        "task": "marketmap.workers.hedge_worker.compute_hedge_edges",
        "schedule": crontab(hour=3, minute=30),
        "options": {"queue": "default"},
    },
    "compute-3d-projection-daily": {
        "task": "marketmap.workers.projection_worker.compute_market_projection_3d",
        "schedule": crontab(hour=4, minute=0),
        "options": {"queue": "default"},
    },
    "compute-local-clusters-daily": {
        "task": "marketmap.workers.local_clustering_worker.compute_local_clusters",
        "schedule": crontab(hour=4, minute=15),
        "options": {"queue": "default"},
    },
    "compute-discovery-clusters-daily": {
        "task": "marketmap.workers.clustering_worker.compute_discovery_clusters",
        "schedule": crontab(hour=4, minute=17),
        "options": {"queue": "default"},
    },
    "compute-projection-distortion-daily": {
        "task": "marketmap.workers.projection_distortion_worker.compute_projection_distortion_scores",
        "schedule": crontab(hour=4, minute=20),
        "options": {"queue": "default"},
    },
    "compute-event-projection-daily": {
        "task": "marketmap.workers.event_projection_worker.run_event_projection_pipeline",
        "schedule": crontab(hour=4, minute=40),
        "options": {"queue": "default"},
    },
    "sync-memgraph-discovery-daily": {
        "task": "marketmap.workers.memgraph_sync_worker.sync_memgraph_discovery",
        "schedule": crontab(hour=4, minute=30),
        "options": {"queue": "default"},
    },
}
