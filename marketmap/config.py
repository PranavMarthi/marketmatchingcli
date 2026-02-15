"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database (Postgres.app PG18 on port 5412)
    database_url: str = (
        "postgresql+asyncpg://postgres@localhost:5412/marketmap"
    )
    database_url_sync: str = (
        "postgresql://postgres@localhost:5412/marketmap"
    )

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"

    # Polymarket
    polymarket_gamma_base_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_base_url: str = "https://clob.polymarket.com"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 256

    # Discovery graph
    discovery_top_k_neighbors: int = 15
    discovery_min_similarity: float = 0.3

    # Hedge graph
    hedge_candidate_semantic_k: int = 20
    hedge_candidate_entity_k: int = 20
    hedge_candidate_category_k: int = 20
    hedge_candidate_time_horizon_days: int = 30
    hedge_candidate_max_total: int = 80
    hedge_price_window_hours: int = 168
    hedge_min_points: int = 6
    hedge_min_confidence: float = 0.35

    # 3D projection (UMAP)
    projection_umap_n_neighbors: int = 25
    projection_umap_min_dist: float = 0.12
    projection_umap_metric: str = "cosine"
    projection_umap_random_state: int = 42
    projection_batch_limit: int = 50000

    # Progressive loading defaults
    discovery_viewport_default_max_nodes: int = 2500
    discovery_viewport_default_max_edges_per_node: int = 15
    discovery_viewport_default_pad: float = 0.15
    discovery_viewport_cache_ttl_seconds: int = 60

    # Worker cadences (seconds)
    market_ingestion_interval_seconds: int = 300  # 5 min
    price_snapshot_interval_seconds: int = 300  # 5 min

    # Polymarket ingestion
    polymarket_page_size: int = 100
    polymarket_max_pages: int = 200  # up to 20k markets

    # Rate limiting
    polymarket_requests_per_second: float = 5.0


settings = Settings()
