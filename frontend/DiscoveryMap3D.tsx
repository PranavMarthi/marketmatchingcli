import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import ForceGraph3D from "react-force-graph-3d";

type Node = {
  id: string;
  label: string;
  category?: string;
  volume?: number;
  x?: number;
  y?: number;
  z?: number;
};

type Link = {
  source: string;
  target: string;
  confidence: number;
  type: string;
};

type GraphPayload = {
  nodes: Node[];
  links: Link[];
  meta: Record<string, unknown>;
};

const API_BASE = "http://127.0.0.1:8000";

function dedupeNodes(nodes: Node[]): Node[] {
  const byId = new Map<string, Node>();
  for (const n of nodes) {
    byId.set(n.id, { ...(byId.get(n.id) || {}), ...n });
  }
  return [...byId.values()];
}

function dedupeLinks(links: Link[]): Link[] {
  const byKey = new Map<string, Link>();
  for (const l of links) {
    const s = typeof l.source === "string" ? l.source : String(l.source);
    const t = typeof l.target === "string" ? l.target : String(l.target);
    const key = `${s}|${t}|${l.type}`;
    const prev = byKey.get(key);
    if (!prev || l.confidence > prev.confidence) {
      byKey.set(key, l);
    }
  }
  return [...byKey.values()];
}

export default function DiscoveryMap3D(): JSX.Element {
  const fgRef = useRef<any>(null);
  const [graph, setGraph] = useState<GraphPayload>({ nodes: [], links: [], meta: {} });
  const [selectedNode, setSelectedNode] = useState<Node | null>(null);
  const [hedgeLinks, setHedgeLinks] = useState<Link[]>([]);
  const [minHedgeConf, setMinHedgeConf] = useState(0.7);

  const refreshViewport = useCallback(async () => {
    const camera = fgRef.current?.camera?.();
    if (!camera) return;

    const scale = Math.max(0.5, Math.min(5, camera.position.length() / 300));
    const span = 8 * scale;

    const minX = camera.position.x - span;
    const maxX = camera.position.x + span;
    const minY = camera.position.y - span;
    const maxY = camera.position.y + span;
    const minZ = camera.position.z - span;
    const maxZ = camera.position.z + span;

    const lod = scale > 2 ? "far" : scale > 1 ? "mid" : "near";
    const url = new URL(`${API_BASE}/graph/discovery/viewport`);
    url.searchParams.set("min_x", `${minX}`);
    url.searchParams.set("max_x", `${maxX}`);
    url.searchParams.set("min_y", `${minY}`);
    url.searchParams.set("max_y", `${maxY}`);
    url.searchParams.set("min_z", `${minZ}`);
    url.searchParams.set("max_z", `${maxZ}`);
    url.searchParams.set("lod", lod);
    url.searchParams.set("max_nodes", "2500");
    url.searchParams.set("max_edges_per_node", lod === "far" ? "4" : lod === "mid" ? "8" : "15");
    url.searchParams.set("min_similarity", "0.55");

    const res = await fetch(url.toString());
    if (!res.ok) return;
    const payload = (await res.json()) as GraphPayload;

    setGraph((prev) => ({
      nodes: dedupeNodes([...prev.nodes, ...payload.nodes]),
      links: dedupeLinks([...prev.links, ...payload.links]),
      meta: payload.meta,
    }));
  }, []);

  useEffect(() => {
    const id = setTimeout(() => {
      void refreshViewport();
    }, 400);
    return () => clearTimeout(id);
  }, [refreshViewport]);

  const onNodeClick = useCallback(
    async (node: Node) => {
      setSelectedNode(node);
      const url = new URL(`${API_BASE}/market/${node.id}/related`);
      url.searchParams.set("edge_types", "hedge");
      url.searchParams.set("min_conf", `${minHedgeConf}`);
      url.searchParams.set("limit", "40");

      const res = await fetch(url.toString());
      if (!res.ok) return;
      const payload = (await res.json()) as GraphPayload;
      setHedgeLinks(payload.links || []);
    },
    [minHedgeConf]
  );

  const mergedLinks = useMemo(() => {
    const overlay = hedgeLinks.map((l) => ({ ...l, type: "hedge" }));
    return dedupeLinks([...graph.links, ...overlay]);
  }, [graph.links, hedgeLinks]);

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#0c1118" }}>
      <div style={{ position: "absolute", zIndex: 10, left: 12, top: 12, color: "#d8e2f0" }}>
        <div>Projection: {String(graph.meta?.projection_version || "none")}</div>
        <div>
          Hedge min conf:
          <input
            type="range"
            min={0.3}
            max={0.95}
            step={0.05}
            value={minHedgeConf}
            onChange={(e) => setMinHedgeConf(Number(e.target.value))}
          />
          {minHedgeConf.toFixed(2)}
        </div>
        {selectedNode && <div>Selected: {selectedNode.label}</div>}
      </div>

      <ForceGraph3D
        ref={fgRef}
        graphData={{ nodes: graph.nodes, links: mergedLinks }}
        nodeId="id"
        nodeLabel={(n: any) => n.label || n.id}
        nodeAutoColorBy="category"
        nodeVal={(n: any) => Math.max(1, Math.log10((n.volume || 1) + 1))}
        linkColor={(l: any) => (l.type === "hedge" ? "#ff6f61" : "#6ec1ff")}
        linkOpacity={0.45}
        onNodeClick={onNodeClick}
        onEngineStop={() => void refreshViewport()}
      />
    </div>
  );
}
