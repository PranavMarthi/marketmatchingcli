import { useEffect, useMemo, useState } from "react";

import PointCloudScene from "./src/components/PointCloudScene";
import { transformPointsForScene, type RawPoint } from "./src/lib/pointsTransform";

type DiscoveryNodePayload = {
  id: string;
  x?: number;
  y?: number;
  z?: number;
};

type DiscoveryPayload = {
  nodes: DiscoveryNodePayload[];
  meta: Record<string, unknown>;
};

const API_BASE = (import.meta.env.VITE_API_URL || "http://127.0.0.1:8000").replace(/\/$/, "");

const DEFAULT_POINT_SIZE = 0.02;

function mapToRawPoint(node: DiscoveryNodePayload): RawPoint {
  return {
    id: node.id,
    x: node.x,
    y: node.y,
    z: node.z
  };
}

export default function DiscoveryMap3D(): JSX.Element {
  const [points, setPoints] = useState<RawPoint[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

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
        pointCount={transformed.pointCount}
        suggestedCameraDistance={transformed.suggestedCameraDistance}
        pointSize={pointSize}
      />
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
