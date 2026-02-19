import { useEffect, useMemo, useState } from "react";

import PointCloudScene from "./src/components/PointCloudScene";
import { transformPointsForScene, type RawPoint } from "./src/lib/pointsTransform";

type DiscoveryNodePayload = {
  id: string;
  label?: string;
  x?: number;
  y?: number;
  z?: number;
  global_x?: number;
  global_y?: number;
  global_z?: number;
  neighborhood_key?: string | null;
  neighborhood_label?: string | null;
  local_cluster_id?: number | null;
  local_distortion?: number | null;
  stitch_distortion?: number | null;
  cluster_id?: string | null;
  distortion_score?: number | null;
};

type DiscoveryPayload = {
  nodes: DiscoveryNodePayload[];
  links?: Array<{
    source: string;
    target: string;
    confidence?: number;
    weight?: number;
    type?: string;
  }>;
  meta?: Record<string, unknown>;
};

const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");
const DEFAULT_POINT_SIZE = 0.26;
const EDGE_MIN_CONF_ATTEMPTS = [0.35, 0.2, 0.1, 0.05] as const;

function mapToRawPoint(node: DiscoveryNodePayload): RawPoint {
  return {
    id: node.id,
    x: node.global_x ?? node.x,
    y: node.global_y ?? node.y,
    z: node.global_z ?? node.z,
    cluster_id: node.cluster_id ?? null,
    neighborhood_key: node.neighborhood_key ?? null,
    neighborhood_label: node.neighborhood_label ?? null,
    local_cluster_id: node.local_cluster_id ?? null,
    local_distortion: node.local_distortion ?? null,
    stitch_distortion: node.stitch_distortion ?? null,
  };
}

export default function DiscoveryMap3D(): JSX.Element {
  const [points, setPoints] = useState<RawPoint[]>([]);
  const [links, setLinks] = useState<
    Array<{ source: string; target: string; confidence?: number; weight?: number; type?: string }>
  >([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [labelById, setLabelById] = useState<Map<string, string>>(new Map());
  const [nodeById, setNodeById] = useState<Map<string, DiscoveryNodePayload>>(new Map());
  const [edgeMinConfUsed, setEdgeMinConfUsed] = useState<number>(EDGE_MIN_CONF_ATTEMPTS[0]);
  const colorBy: "neighborhood" = "neighborhood";
  const intraClusterScale = 1.25;
  const macroSeparation = 1.22;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    const load = async () => {
      try {
        let payload: DiscoveryPayload | null = null;

        for (const minConf of EDGE_MIN_CONF_ATTEMPTS) {
          const url = new URL(`${API_BASE}/graph/discovery/all`);
          url.searchParams.set("include_edges", "true");
          url.searchParams.set("include_local", "true");
          url.searchParams.set("entity", "events");
          url.searchParams.set("min_conf", String(minConf));

          const res = await fetch(url.toString(), { signal: controller.signal });
          if (!res.ok) {
            throw new Error(`request failed (${res.status})`);
          }

          const nextPayload = (await res.json()) as DiscoveryPayload;
          payload = nextPayload;
          setEdgeMinConfUsed(minConf);

          const linkCount = Array.isArray(nextPayload.links) ? nextPayload.links.length : 0;
          if (linkCount > 0) break;
        }

        if (!payload) throw new Error("empty payload");
        const rawPoints = (payload.nodes || []).map(mapToRawPoint);
        setPoints(rawPoints);
        setLinks(Array.isArray(payload.links) ? payload.links : []);

        const labels = new Map<string, string>();
        const nodes = new Map<string, DiscoveryNodePayload>();
        for (const node of payload.nodes || []) {
          labels.set(node.id, node.label || node.id);
          nodes.set(node.id, node);
        }
        setLabelById(labels);
        setNodeById(nodes);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        setError(err instanceof Error ? err.message : "unknown fetch error");
      } finally {
        setLoading(false);
      }
    };

    void load();
    return () => controller.abort();
  }, []);

  const transformed = useMemo(
    () =>
      transformPointsForScene(points, {
        colorBy,
        intraClusterScale,
        macroSeparation,
      }),
    [points, colorBy, intraClusterScale, macroSeparation]
  );

  const pointSize = DEFAULT_POINT_SIZE;

  const sceneEdges = useMemo(() => {
    if (!links.length || transformed.validPoints.length === 0) return [];

    const idToIndex = new Map<string, number>();
    for (let i = 0; i < transformed.validPoints.length; i += 1) {
      idToIndex.set(transformed.validPoints[i].id, i);
    }

    const out: Array<{ sourceIndex: number; targetIndex: number; confidence: number }> = [];
    const seen = new Set<string>();
    for (const edge of links) {
      const a = idToIndex.get(edge.source);
      const b = idToIndex.get(edge.target);
      if (a === undefined || b === undefined || a === b) continue;

      const lo = a < b ? a : b;
      const hi = a < b ? b : a;
      const key = `${lo}:${hi}`;
      if (seen.has(key)) continue;
      seen.add(key);

      const confidenceRaw = edge.confidence ?? edge.weight ?? 0;
      const confidence = Number.isFinite(confidenceRaw) ? Math.max(0, Math.min(1, confidenceRaw)) : 0;
      if (confidence <= 0) continue;

      out.push({ sourceIndex: a, targetIndex: b, confidence });
      if (out.length >= 24000) break;
    }
    return out;
  }, [links, transformed.validPoints]);

  const handlePointClick = (pointIndex: number) => {
    const point = transformed.validPoints[pointIndex];
    if (!point) return;
    setSelectedId((prev) => (prev === point.id ? null : point.id));
  };
  const selectedQuestion = selectedId ? labelById.get(selectedId) ?? selectedId : null;

  const selectedRelated = useMemo(() => {
    if (!selectedId) return [] as Array<{ id: string; label: string; confidence: number }>;
    const list: Array<{ id: string; label: string; confidence: number }> = [];
    for (const edge of links) {
      const isSource = edge.source === selectedId;
      const isTarget = edge.target === selectedId;
      if (!isSource && !isTarget) continue;
      const otherId = isSource ? edge.target : edge.source;
      const confidenceRaw = edge.confidence ?? edge.weight ?? 0;
      const confidence = Number.isFinite(confidenceRaw) ? Math.max(0, Math.min(1, confidenceRaw)) : 0;
      list.push({ id: otherId, label: labelById.get(otherId) ?? otherId, confidence });
    }
    list.sort((a, b) => b.confidence - a.confidence || a.label.localeCompare(b.label));

    const deduped: Array<{ id: string; label: string; confidence: number }> = [];
    const seen = new Set<string>();
    for (const item of list) {
      if (seen.has(item.id)) continue;
      seen.add(item.id);
      deduped.push(item);
      if (deduped.length >= 12) break;
    }
    return deduped;
  }, [selectedId, links, labelById]);

  const selectedNode = selectedId ? nodeById.get(selectedId) ?? null : null;

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", background: "#05060a" }}>
      <div
        style={{
          position: "absolute",
          top: 24,
          left: 24,
          zIndex: 20,
          color: "#eef3ff",
          fontSize: 24,
          letterSpacing: "0.04em",
          fontFamily:
            '"GT Standard Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
          textShadow: "0 0 14px rgba(210, 225, 255, 0.35)",
        }}
      >
        PolySpace
      </div>

      <PointCloudScene
        positions={transformed.positions}
        colors={transformed.colors}
        pointCount={transformed.pointCount}
        edges={sceneEdges}
        neighborhoodShells={[]}
        suggestedCameraDistance={transformed.suggestedCameraDistance}
        pointSize={pointSize}
        onPointClick={handlePointClick}
      />

      {selectedQuestion && (
        <div
          style={{
            position: "absolute",
            right: 24,
            top: 86,
            width: "min(780px, calc(100vw - 48px))",
            maxHeight: "72vh",
            overflowY: "auto",
            padding: "18px 20px",
            borderRadius: 14,
            background: "linear-gradient(180deg, rgba(7,11,19,0.93), rgba(7,12,20,0.86))",
            border: "1px solid rgba(146, 174, 221, 0.28)",
            boxShadow: "0 10px 34px rgba(0, 0, 0, 0.35)",
            color: "#e9f0ff",
            fontSize: 16,
            lineHeight: 1.45,
            zIndex: 30,
          }}
        >
          <div style={{ fontSize: 42, lineHeight: 1, marginBottom: 10 }}>?</div>
          <div
            style={{
              fontSize: 40,
              fontWeight: 600,
              lineHeight: 1.14,
              marginBottom: 14,
              color: "#f2f6ff",
            }}
          >
            {selectedQuestion}
          </div>
          <div style={{ opacity: 0.8, fontSize: 18, marginBottom: 12 }}>{selectedId}</div>
          <div style={{ opacity: 0.9, fontSize: 20, marginBottom: 18 }}>
            {`tags: ${selectedNode?.neighborhood_label || selectedNode?.neighborhood_key || "none"}`}
          </div>

          <div style={{ fontSize: 30, marginBottom: 10, color: "#eaf2ff" }}>related nodes:</div>
          <div style={{ display: "grid", gap: 6 }}>
            {selectedRelated.length === 0 ? (
              <div style={{ opacity: 0.75 }}>none</div>
            ) : (
              selectedRelated.map((item) => (
                <div key={item.id} style={{ fontSize: 36, lineHeight: 1.12, color: "#f4f7ff" }}>
                  {item.label}
                </div>
              ))
            )}
          </div>
        </div>
      )}

      {(loading || error) && (
        <div
          style={{
            position: "absolute",
            top: 24,
            right: 24,
            color: "#95a6c8",
            fontSize: 12,
            opacity: 0.8,
          }}
        >
          {loading ? "loading points..." : `error: ${error}`}
        </div>
      )}

      {!loading && !error && (
        <div
          style={{
            position: "absolute",
            top: 52,
            right: 24,
            color: "#95a6c8",
            fontSize: 11,
            opacity: 0.9,
          }}
        >
          {`nodes: ${transformed.pointCount} • edges: ${sceneEdges.length} (min_conf=${edgeMinConfUsed})`}
        </div>
      )}
    </div>
  );
}
