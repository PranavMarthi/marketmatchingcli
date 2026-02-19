"""Worker wrapper for local neighborhood clustering."""

from marketmap.clustering.local_clusters import compute_local_clusters
from marketmap.workers.celery_app import app


@app.task(bind=True, name="marketmap.workers.local_clustering_worker.compute_local_clusters", max_retries=2)
def compute_local_clusters_task(self) -> dict:  # type: ignore[type-arg]
    return compute_local_clusters()
