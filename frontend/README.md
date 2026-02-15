# React Force Graph 3D (first pass)

This folder contains a first-pass `React Force Graph 3D` component that uses progressive loading and hedge overlays:

- `DiscoveryMap3D.tsx`

## What it does

- Calls `GET /graph/discovery/viewport` using camera-driven bounds.
- Uses LOD (`near/mid/far`) to tune edge density.
- Merges paged viewport payloads into a local graph store.
- On node click, calls `GET /market/{id}/related?edge_types=hedge&min_conf=...` and overlays hedge links.

## Integrate in your React app

1. Install dependencies in your frontend project:
   - `react-force-graph-3d`
   - `three`
2. Copy `DiscoveryMap3D.tsx` into your React `src/` tree.
3. Render `<DiscoveryMap3D />` from your app.
4. Ensure backend API is running at `http://127.0.0.1:8000` (or update `API_BASE`).

This is intentionally a first pass and can be expanded with frustum math, tile eviction, and animation polish.
