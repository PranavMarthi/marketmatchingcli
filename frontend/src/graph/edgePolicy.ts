export type DiscoveryEdgeLike = {
  source: string;
  target: string;
  confidence?: number;
  weight?: number;
  type?: string;
};

export type FilteredEdge<T extends DiscoveryEdgeLike> = T & {
  isBridgeEdge: boolean;
  __weight: number;
};

export type EdgePolicyOptions = {
  maxEdgesBudget?: number;
  maxBridgeEdgesPerNode?: number;
  bridgeMinPercentile?: number;
};

export type EdgePolicyResult<T extends DiscoveryEdgeLike> = {
  edges: FilteredEdge<T>[];
  keptIntra: number;
  keptBridge: number;
  dropped: number;
};

const DEFAULT_BUDGET = 150000;
const DEFAULT_MAX_BRIDGES_PER_NODE = 1;
const DEFAULT_BRIDGE_MIN_PERCENTILE = 99;

function edgeWeight(edge: DiscoveryEdgeLike): number {
  const raw = edge.weight ?? edge.confidence ?? 0;
  return Number.isFinite(raw) ? Math.max(0, Number(raw)) : 0;
}

function sourceId(edge: DiscoveryEdgeLike): string {
  return typeof edge.source === "string" ? edge.source : String(edge.source);
}

function targetId(edge: DiscoveryEdgeLike): string {
  return typeof edge.target === "string" ? edge.target : String(edge.target);
}

function percentileValue(values: number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  const idx = Math.min(sorted.length - 1, Math.max(0, Math.floor((p / 100) * sorted.length)));
  return sorted[idx] ?? 0;
}

export function applyDiscoveryEdgePolicy<T extends DiscoveryEdgeLike>(
  edges: T[],
  clusterByNodeId: Map<string, number>,
  options: EdgePolicyOptions = {}
): EdgePolicyResult<T> {
  const maxEdgesBudget = options.maxEdgesBudget ?? DEFAULT_BUDGET;
  const maxBridgeEdgesPerNode = options.maxBridgeEdgesPerNode ?? DEFAULT_MAX_BRIDGES_PER_NODE;
  const bridgeMinPercentile = options.bridgeMinPercentile ?? DEFAULT_BRIDGE_MIN_PERCENTILE;

  const intra: FilteredEdge<T>[] = [];
  const cross: FilteredEdge<T>[] = [];

  for (const edge of edges) {
    const src = sourceId(edge);
    const tgt = targetId(edge);
    const srcCluster = clusterByNodeId.get(src);
    const tgtCluster = clusterByNodeId.get(tgt);
    const weighted = { ...edge, isBridgeEdge: false, __weight: edgeWeight(edge) };

    if (srcCluster !== undefined && tgtCluster !== undefined && srcCluster === tgtCluster) {
      intra.push(weighted);
    } else {
      cross.push(weighted);
    }
  }

  intra.sort((a, b) => b.__weight - a.__weight);
  cross.sort((a, b) => b.__weight - a.__weight);

  const bridgeThreshold = percentileValue(
    cross.map((edge) => edge.__weight),
    bridgeMinPercentile
  );

  const perNodeBridgeCount = new Map<string, number>();
  const keptBridge: FilteredEdge<T>[] = [];

  for (const edge of cross) {
    if (edge.__weight < bridgeThreshold) {
      continue;
    }

    const src = sourceId(edge);
    const tgt = targetId(edge);
    const srcCount = perNodeBridgeCount.get(src) ?? 0;
    const tgtCount = perNodeBridgeCount.get(tgt) ?? 0;
    if (srcCount >= maxBridgeEdgesPerNode || tgtCount >= maxBridgeEdgesPerNode) {
      continue;
    }

    perNodeBridgeCount.set(src, srcCount + 1);
    perNodeBridgeCount.set(tgt, tgtCount + 1);
    keptBridge.push({ ...edge, isBridgeEdge: true });
  }

  const chosen: FilteredEdge<T>[] = [];
  for (const edge of intra) {
    if (chosen.length >= maxEdgesBudget) break;
    chosen.push(edge);
  }
  for (const edge of keptBridge) {
    if (chosen.length >= maxEdgesBudget) break;
    chosen.push(edge);
  }

  return {
    edges: chosen,
    keptIntra: Math.min(intra.length, maxEdgesBudget),
    keptBridge: Math.max(0, chosen.length - Math.min(intra.length, maxEdgesBudget)),
    dropped: Math.max(0, edges.length - chosen.length)
  };
}
