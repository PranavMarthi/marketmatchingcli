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
    # Milestone 3: Hedge graph workers (daily, after fresh price snapshots)
    "compute-hedge-edges-daily": {
        "task": "marketmap.workers.hedge_worker.compute_hedge_edges",
        "schedule": crontab(hour=3, minute=30),
        "options": {"queue": "default"},
    },
}
