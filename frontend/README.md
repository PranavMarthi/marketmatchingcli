# MarketMap 3D Frontend (React Force Graph 3D)

This frontend renders the full semantic discovery graph (all nodes and edges) and hedge overlays from:

- `GET /graph/discovery/all`
- `GET /market/{id}/related` (hedge overlay on click)

The component is in `frontend/DiscoveryMap3D.tsx` and is wired into a runnable Vite app.

## Prerequisites

- Node.js `>=18`
- npm `>=9`
- Python `3.11+` (project currently uses `.venv` with `3.14`)
- Postgres running on `localhost:5412`
- Redis running on `localhost:6379`
- Memgraph running on `localhost:7687`

## 1) Backend setup

From repo root (`marketmap`):

```bash
source .venv/bin/activate
pip install -e .
alembic upgrade head
```

## 1b) Memgraph setup

Run Memgraph locally (example via Docker):

```bash
docker run -it --rm -p 7687:7687 memgraph/memgraph-platform
```

Optional `.env` settings:

```bash
MEMGRAPH_ENABLED=true
MEMGRAPH_URI=bolt://localhost:7687
MEMGRAPH_USERNAME=
MEMGRAPH_PASSWORD=
```

## 2) Start backend API

```bash
source .venv/bin/activate
uvicorn marketmap.api.main:app --host 127.0.0.1 --port 8000 --reload
```

Open API docs at `http://127.0.0.1:8000/docs`.

## 3) Start Celery worker + beat (separate terminals)

Worker:

```bash
source .venv/bin/activate
celery -A marketmap.workers.celery_app.app worker -l info
```

Beat scheduler:

```bash
source .venv/bin/activate
celery -A marketmap.workers.celery_app.app beat -l info
```

## 4) Ensure projection data + Memgraph graph exist

Run once if needed (required for `x,y,z` map rendering):

```bash
source .venv/bin/activate
python -c "from marketmap.workers.projection_worker import compute_market_projection_3d; print(compute_market_projection_3d(batch_limit=20500))"
```

Sync Postgres discovery graph into Memgraph:

```bash
source .venv/bin/activate
python -c "from marketmap.workers.memgraph_sync_worker import sync_memgraph_discovery; print(sync_memgraph_discovery(min_conf=0.3))"
```

## 5) Start frontend

From `frontend/`:

```bash
npm install
npm run dev
```

Open `http://127.0.0.1:5173`.

## Environment variables

Frontend API base URL (optional):

- `VITE_API_URL` (default: `http://127.0.0.1:8000`)

Example:

```bash
VITE_API_URL=http://127.0.0.1:8000 npm run dev
```

## How to verify full graph rendering

1. Open browser DevTools Network tab.
2. Confirm one request to `/graph/discovery/all?min_conf=0.3&include_edges=true`.
3. Confirm graph node/edge counts appear in the HUD.
4. Click a node and confirm `/market/{id}/related?edge_types=hedge...` is requested.

## Troubleshooting

- **`{"detail":"Not Found"}` for endpoint**
  - Restart API (`uvicorn ... --reload`) and confirm latest code is running.

- **CORS errors**
  - Backend currently allows all origins. Ensure frontend calls `127.0.0.1:8000` (not mixed hostnames).

- **Empty graph (no nodes)**
  - Ensure projection table is populated (`market_projection_3d` rows exist).
  - Ensure Memgraph has data (run sync command in step 4).
  - Verify `/graph/discovery/all?include_edges=false` returns nodes.

- **Frontend loads but no all-graph data**
  - Check Network tab for `/graph/discovery/all` response status.
  - Confirm Memgraph is running on port `7687`.
  - If Memgraph is down, API falls back to Postgres for `/graph/discovery/all`.
