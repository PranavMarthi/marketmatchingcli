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

type EventContextPayload = {
  event_id: string;
  label: string;
  tags: string[];
  related_nodes: Array<{
    id: string;
    label: string;
    distance?: number | null;
  }>;
};

const defaultApiBase = `${window.location.protocol}//${window.location.hostname}:8000`;
const API_BASE = (import.meta.env.VITE_API_URL || defaultApiBase).replace(/\/$/, "");
const DEFAULT_POINT_SIZE = 0.187;

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
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [labelById, setLabelById] = useState<Map<string, string>>(new Map());
  const [eventContext, setEventContext] = useState<EventContextPayload | null>(null);
  const [eventContextLoading, setEventContextLoading] = useState(false);
  const colorBy: "neighborhood" = "neighborhood";
  const intraClusterScale = 1.25;
  const macroSeparation = 1.22;

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    const load = async () => {
      try {
        const url = new URL(`${API_BASE}/graph/discovery/all`);
        url.searchParams.set("include_edges", "true");
        url.searchParams.set("include_local", "true");
        url.searchParams.set("entity", "events");
        url.searchParams.set("min_conf", "0.35");

        const res = await fetch(url.toString(), { signal: controller.signal });
        if (!res.ok) {
          throw new Error(`request failed (${res.status})`);
        }

        const payload = (await res.json()) as DiscoveryPayload;
        const rawPoints = (payload.nodes || []).map(mapToRawPoint);
        setPoints(rawPoints);
        setLinks(Array.isArray(payload.links) ? payload.links : []);

        const labels = new Map<string, string>();
        for (const node of payload.nodes || []) {
          labels.set(node.id, node.label || node.id);
        }
        setLabelById(labels);
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

  const focusId = hoveredId ?? selectedId;

  const focusEdges = useMemo(() => {
    if (!focusId || !links.length || transformed.validPoints.length === 0) return [];

    const idToIndex = new Map<string, number>();
    for (let i = 0; i < transformed.validPoints.length; i += 1) {
      idToIndex.set(transformed.validPoints[i].id, i);
    }

    const out: Array<{ sourceIndex: number; targetIndex: number; confidence: number }> = [];
    const seen = new Set<string>();
    for (const edge of links) {
      if (edge.source !== focusId && edge.target !== focusId) continue;
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
      out.push({ sourceIndex: a, targetIndex: b, confidence: Math.max(confidence, 0.9) });
      if (out.length >= 400) break;
    }
    return out;
  }, [focusId, links, transformed.validPoints]);

  const mergedEdges = useMemo(() => {
    if (!focusEdges.length) return sceneEdges;
    const byKey = new Map<string, { sourceIndex: number; targetIndex: number; confidence: number }>();
    for (const edge of sceneEdges) {
      const lo = edge.sourceIndex < edge.targetIndex ? edge.sourceIndex : edge.targetIndex;
      const hi = edge.sourceIndex < edge.targetIndex ? edge.targetIndex : edge.sourceIndex;
      byKey.set(`${lo}:${hi}`, edge);
    }
    for (const edge of focusEdges) {
      const lo = edge.sourceIndex < edge.targetIndex ? edge.sourceIndex : edge.targetIndex;
      const hi = edge.sourceIndex < edge.targetIndex ? edge.targetIndex : edge.sourceIndex;
      const key = `${lo}:${hi}`;
      const existing = byKey.get(key);
      if (!existing || edge.confidence > existing.confidence) {
        byKey.set(key, edge);
      }
    }
    return Array.from(byKey.values());
  }, [sceneEdges, focusEdges]);

  const highlightedPointIndices = useMemo(() => {
    if (!selectedId || transformed.validPoints.length === 0) return null;
    const idToIndex = new Map<string, number>();
    for (let i = 0; i < transformed.validPoints.length; i += 1) {
      idToIndex.set(transformed.validPoints[i].id, i);
    }

    const selectedIdx = idToIndex.get(selectedId);
    if (selectedIdx === undefined) return null;

    const out = new Set<number>();
    out.add(selectedIdx);
    for (const edge of links) {
      if (edge.source === selectedId) {
        const idx = idToIndex.get(edge.target);
        if (idx !== undefined) out.add(idx);
      } else if (edge.target === selectedId) {
        const idx = idToIndex.get(edge.source);
        if (idx !== undefined) out.add(idx);
      }
    }
    if (eventContext && eventContext.event_id === selectedId) {
      for (const related of eventContext.related_nodes) {
        const idx = idToIndex.get(related.id);
        if (idx !== undefined) out.add(idx);
      }
    }
    return out;
  }, [selectedId, transformed.validPoints, links, eventContext]);

  const selectedPointIndex = useMemo(() => {
    if (!selectedId) return null;
    for (let i = 0; i < transformed.validPoints.length; i += 1) {
      if (transformed.validPoints[i].id === selectedId) return i;
    }
    return null;
  }, [selectedId, transformed.validPoints]);

  const handlePointClick = (pointIndex: number) => {
    const point = transformed.validPoints[pointIndex];
    if (!point) return;
    setSelectedId((prev) => (prev === point.id ? null : point.id));
  };
  const handlePointHover = (pointIndex: number | null) => {
    if (pointIndex === null) {
      setHoveredId(null);
      return;
    }
    const point = transformed.validPoints[pointIndex];
    setHoveredId(point ? point.id : null);
  };
  const selectedQuestion = selectedId ? labelById.get(selectedId) ?? selectedId : null;

  useEffect(() => {
    if (!selectedId) {
      setEventContext(null);
      setEventContextLoading(false);
      return;
    }

    const controller = new AbortController();
    setEventContext(null);
    setEventContextLoading(true);

    const loadContext = async () => {
      try {
        const url = new URL(`${API_BASE}/event/${encodeURIComponent(selectedId)}/context`);
        url.searchParams.set("related_limit", "5000");
        const res = await fetch(url.toString(), { signal: controller.signal });
        if (!res.ok) {
          throw new Error(`context request failed (${res.status})`);
        }
        const payload = (await res.json()) as EventContextPayload;
        setEventContext(payload);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") return;
        setEventContext(null);
      } finally {
        setEventContextLoading(false);
      }
    };

    void loadContext();
    return () => controller.abort();
  }, [selectedId]);

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
        edges={mergedEdges}
        highlightedPointIndices={highlightedPointIndices}
        selectedPointIndex={selectedPointIndex}
        neighborhoodShells={[]}
        suggestedCameraDistance={transformed.suggestedCameraDistance}
        pointSize={pointSize}
        onPointClick={handlePointClick}
        onPointHover={handlePointHover}
      />

      {selectedQuestion && (
        <div
          style={{
            position: "absolute",
            left: 24,
            bottom: 24,
            maxWidth: 760,
            padding: "9px 12px",
            borderRadius: 8,
            background: "rgba(8, 12, 20, 0.84)",
            border: "1px solid rgba(170, 190, 230, 0.22)",
            color: "#e9f0ff",
            fontSize: 12,
            lineHeight: 1.35,
            zIndex: 30,
          }}
        >
          <div style={{ fontWeight: 700, marginBottom: 6 }}>{selectedQuestion}</div>
          <div style={{ opacity: 0.8, fontSize: 11, marginBottom: 6 }}>{selectedId}</div>
          <div style={{ marginBottom: 6 }}>
            <span style={{ opacity: 0.85 }}>tags:</span>{" "}
            {eventContextLoading
              ? "loading..."
              : eventContext && eventContext.tags.length > 0
                ? eventContext.tags.join(", ")
                : "none"}
          </div>
          <div>
            <span style={{ opacity: 0.85 }}>related nodes:</span>
            {eventContextLoading ? (
              <div style={{ marginTop: 4, opacity: 0.8 }}>loading...</div>
            ) : eventContext && eventContext.related_nodes.length > 0 ? (
              <div style={{ marginTop: 4, maxHeight: 180, overflow: "auto" }}>
                {eventContext.related_nodes.map((n) => (
                  <div key={n.id} style={{ marginBottom: 2 }}>
                    {n.label}
                  </div>
                ))}
              </div>
            ) : (
              <div style={{ marginTop: 4, opacity: 0.8 }}>none</div>
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
    </div>
  );
}
