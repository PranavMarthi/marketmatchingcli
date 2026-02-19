"""Application configuration via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # Database (Postgres.app PG18 on port 5412)
    database_url: str = "postgresql+asyncpg://postgres@localhost:5412/marketmap"
    database_url_sync: str = "postgresql://postgres@localhost:5412/marketmap"

    # Redis
    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "redis://localhost:6379/1"

    # Memgraph
    memgraph_enabled: bool = True
    memgraph_uri: str = "bolt://localhost:7687"
    memgraph_username: str = ""
    memgraph_password: str = ""

    # Polymarket
    polymarket_gamma_base_url: str = "https://gamma-api.polymarket.com"
    polymarket_clob_base_url: str = "https://clob.polymarket.com"
    polymarket_events_json_path: str = "all_active_polymarket_events.json"

    # Embeddings
    openai_api_key: str = ""
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    embedding_dim: int = 1024
    embedding_batch_size: int = 256
    embedding_use_domain_hints: bool = True

    # Discovery graph
    discovery_top_k_neighbors: int = 15
    discovery_min_similarity: float = 0.3
    discovery_cross_neighborhood_edges_enabled: bool = False
    discovery_cross_neighborhood_top_k: int = 2
    discovery_cross_neighborhood_min_similarity: float = 0.72

    # Hedge graph
    hedge_candidate_semantic_k: int = 20
    hedge_candidate_entity_k: int = 20
    hedge_candidate_category_k: int = 20
    hedge_candidate_time_horizon_days: int = 30
    hedge_candidate_max_total: int = 80
    hedge_price_window_hours: int = 168
    hedge_min_points: int = 6
    hedge_min_confidence: float = 0.35

    # 3D projection (two-stage UMAP)
    projection_umap_random_state: int = 42
    projection_stage1_n_components: int = 15
    projection_stage1_n_neighbors: int = 120
    projection_stage1_min_dist: float = 0.05
    projection_stage1_metric: str = "cosine"
    projection_stage1_init: str = "spectral"
    projection_stage2_n_components: int = 3
    projection_stage2_n_neighbors: int = 50
    projection_stage2_min_dist: float = 0.12
    projection_stage2_metric: str = "euclidean"
    projection_stage2_init: str = "random"
    projection_smoothing_enabled: bool = True
    projection_smoothing_iterations: int = 2
    projection_smoothing_k_neighbors: int = 24
    projection_smoothing_distortion_threshold: float = 0.9
    projection_smoothing_alpha: float = 0.18
    projection_outlier_guard_enabled: bool = True
    projection_outlier_radius_quantile: float = 0.985
    projection_outlier_guard_alpha: float = 0.22
    projection_outlier_kmeans_clusters: int = 24
    projection_batch_limit: int = 50000

    # Hierarchical projection / neighborhoods
    neighborhood_min_size: int = 80
    neighborhood_small_merge_threshold: int = 50
    neighborhood_local_umap_n_neighbors: int = 30
    neighborhood_local_umap_min_dist: float = 0.10
    neighborhood_local_umap_metric: str = "cosine"
    neighborhood_local_umap_seed: int = 42
    neighborhood_stitch_scale_min: float = 0.5
    neighborhood_stitch_scale_max: float = 2.5
    neighborhood_local_cluster_min_size: int = 120

    # Progressive loading defaults
    discovery_viewport_default_max_nodes: int = 2500
    discovery_viewport_default_max_edges_per_node: int = 15
    discovery_viewport_default_pad: float = 0.15
    discovery_viewport_cache_ttl_seconds: int = 60
    discovery_cluster_min_confidence: float = 0.35
    discovery_cluster_resolution: float = 0.8
    discovery_cluster_seed: int = 1337
    discovery_cluster_target_main_clusters: int = 20

    # Projection distortion scoring
    projection_distortion_k_neighbors: int = 10

    # Worker cadences (seconds)
    market_ingestion_interval_seconds: int = 300  # 5 min
    price_snapshot_interval_seconds: int = 300  # 5 min

    # Polymarket ingestion
    polymarket_page_size: int = 100
    polymarket_max_pages: int = 200  # up to 20k markets

    # Rate limiting
    polymarket_requests_per_second: float = 5.0


settings = Settings()
