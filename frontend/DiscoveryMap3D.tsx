import { useEffect, useMemo, useState } from "react";

import PointCloudScene from "./src/components/PointCloudScene";
import { transformPointsForScene, type RawPoint } from "./src/lib/pointsTransform";

type DiscoveryNodePayload = {
  id: string;
  label?: string;
  x?: number;
  y?: number;
  z?: number;
  cluster_id?: string | null;
  distortion_score?: number | null;
};

type DiscoveryPayload = {
  nodes: DiscoveryNodePayload[];
  meta?: Record<string, unknown>;
};

const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const DEFAULT_POINT_SIZE = 0.187;
type PointDatum = RawPoint;

function mapToRawPoint(node: DiscoveryNodePayload): RawPoint {
  return {
    id: node.id,
    x: node.x,
    y: node.y,
    z: node.z,
    cluster_id: node.cluster_id ?? null,
  };
}

export default function DiscoveryMap3D(): JSX.Element {
  const [points, setPoints] = useState<PointDatum[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [hoveredId, setHoveredId] = useState<string | null>(null);
  const [selectedQuestion, setSelectedQuestion] = useState<string | null>(null);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [labelById, setLabelById] = useState<Map<string, string>>(new Map());
  const [distortionById, setDistortionById] = useState<Map<string, number | null>>(new Map());

  useEffect(() => {
    const controller = new AbortController();
    setLoading(true);
    setError(null);

    const load = async () => {
      try {
        const url = new URL(`${API_BASE}/graph/discovery/all`);
        url.searchParams.set("include_edges", "false");

        const res = await fetch(url.toString(), { signal: controller.signal });
        if (!res.ok) {
          throw new Error(`request failed (${res.status})`);
        }

        const payload = (await res.json()) as DiscoveryPayload;
        const rawPoints = (payload.nodes || []).map(mapToRawPoint);
        setPoints(rawPoints);
        const labels = new Map<string, string>();
        const distortions = new Map<string, number | null>();
        for (const node of payload.nodes || []) {
          labels.set(node.id, node.label || node.id);
          distortions.set(
            node.id,
            typeof node.distortion_score === "number" ? node.distortion_score : null
          );
        }
        setLabelById(labels);
        setDistortionById(distortions);
      } catch (err) {
        if (err instanceof DOMException && err.name === "AbortError") {
          return;
        }
        console.error("point cloud fetch failed", err);
        setError(err instanceof Error ? err.message : "unknown fetch error");
      } finally {
        setLoading(false);
      }
    };

    void load();
    return () => controller.abort();
  }, []);

  const transformed = useMemo(() => transformPointsForScene(points), [points]);
  const pointSize = DEFAULT_POINT_SIZE;

  const handlePointClick = (pointIndex: number) => {
    const point = transformed.validPoints[pointIndex];
    if (!point) return;
    setSelectedId(point.id);
    setSelectedQuestion(labelById.get(point.id) || point.id);
  };

  const handlePointHover = (pointIndex: number | null) => {
    if (pointIndex === null) {
      setHoveredId(null);
      return;
    }
    const point = transformed.validPoints[pointIndex];
    if (!point) {
      setHoveredId(null);
      return;
    }
    setHoveredId(point.id);
  };

  const hoveredQuestion = hoveredId ? labelById.get(hoveredId) || hoveredId : null;
  const hoveredDistortion = hoveredId ? distortionById.get(hoveredId) ?? null : null;
  const selectedDistortion = selectedId ? distortionById.get(selectedId) ?? null : null;

  return (
    <div style={{ width: "100vw", height: "100vh", position: "relative", background: "#05060a" }}>
      <div
        style={{
          position: "absolute",
          top: 48,
          left: 56,
          zIndex: 20,
          color: "#eef3ff",
          fontSize: 24,
          letterSpacing: "0.04em",
          fontFamily:
            '"GT Standard Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
          textShadow: "0 0 14px rgba(210, 225, 255, 0.35)",
          pointerEvents: "none"
        }}
      >
        PolySpace
      </div>

      <PointCloudScene
        positions={transformed.positions}
        colors={transformed.colors}
        pointCount={transformed.pointCount}
        suggestedCameraDistance={transformed.suggestedCameraDistance}
        pointSize={pointSize}
        onPointClick={handlePointClick}
        onPointHover={handlePointHover}
      />
      {hoveredQuestion && (
        <div
          style={{
            position: "absolute",
            left: 56,
            top: 96,
            maxWidth: 520,
            padding: "7px 10px",
            borderRadius: 8,
            background: "rgba(8, 12, 20, 0.84)",
            border: "1px solid rgba(170, 190, 230, 0.22)",
            color: "#e9f0ff",
            fontFamily:
              '"GT Standard Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            fontSize: 11,
            lineHeight: 1.35,
            zIndex: 30,
            pointerEvents: "none",
          }}
        >
          <div>{hoveredQuestion}</div>
          {hoveredDistortion !== null && (
            <div style={{ opacity: 0.72, marginTop: 4 }}>
              distortion: {hoveredDistortion.toFixed(3)}
            </div>
          )}
        </div>
      )}
      {selectedQuestion && (
        <div
          style={{
            position: "absolute",
            right: 24,
            top: 24,
            maxWidth: 420,
            padding: "10px 12px",
            borderRadius: 10,
            background: "rgba(8, 12, 20, 0.88)",
            border: "1px solid rgba(170, 190, 230, 0.24)",
            color: "#e8efff",
            fontFamily:
              '"GT Standard Mono", "SFMono-Regular", Menlo, Monaco, Consolas, "Liberation Mono", "Courier New", monospace',
            fontSize: 12,
            lineHeight: 1.35,
            zIndex: 30,
          }}
        >
          <div style={{ opacity: 0.72, marginBottom: 6 }}>Selected market</div>
          <div style={{ marginBottom: 6 }}>{selectedQuestion}</div>
          {selectedDistortion !== null && (
            <div style={{ opacity: 0.72, marginBottom: 6 }}>
              distortion: {selectedDistortion.toFixed(3)}
            </div>
          )}
          {selectedId && <div style={{ opacity: 0.65 }}>id: {selectedId}</div>}
        </div>
      )}
      {(loading || error) && (
        <div
          style={{
            position: "absolute",
            bottom: 12,
            left: 12,
            color: "#95a6c8",
            fontSize: 12,
            opacity: 0.72,
            pointerEvents: "none"
          }}
        >
          {loading ? "loading points..." : `error: ${error}`}
        </div>
      )}
    </div>
  );
}
