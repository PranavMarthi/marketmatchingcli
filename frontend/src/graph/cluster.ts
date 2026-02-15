type PointNode = {
  id: string;
  x?: number;
  y?: number;
  z?: number;
};

export type ClusterOptions = {
  targetClusterCount?: number;
  maxIterations?: number;
  seed?: number;
  projectionVersion?: string;
};

export type ClusterResult = {
  clusterByNodeId: Map<string, number>;
  clusterCount: number;
  cacheKey: string;
};

type Vec3 = { x: number; y: number; z: number };

const DEFAULT_TARGET_CLUSTER_COUNT = 12;
const DEFAULT_MAX_ITERATIONS = 10;
const DEFAULT_SEED = 1337;

const cache = new Map<string, ClusterResult>();

function finiteCoord(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

function checksumNodes(nodes: PointNode[]): number {
  let hash = 2166136261;
  const stride = Math.max(1, Math.floor(nodes.length / 256));
  for (let i = 0; i < nodes.length; i += stride) {
    const node = nodes[i];
    const x = finiteCoord(node.x) ? Math.round(node.x * 1000) : 0;
    const y = finiteCoord(node.y) ? Math.round(node.y * 1000) : 0;
    const z = finiteCoord(node.z) ? Math.round(node.z * 1000) : 0;
    hash ^= x + (y << 1) + (z << 2) + node.id.length;
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function keyForNodes(nodes: PointNode[], options: ClusterOptions): string {
  return [
    options.projectionVersion || "none",
    nodes.length,
    checksumNodes(nodes),
    options.targetClusterCount ?? DEFAULT_TARGET_CLUSTER_COUNT,
    options.maxIterations ?? DEFAULT_MAX_ITERATIONS,
    options.seed ?? DEFAULT_SEED
  ].join(":");
}

function distanceSq(a: Vec3, b: Vec3): number {
  const dx = a.x - b.x;
  const dy = a.y - b.y;
  const dz = a.z - b.z;
  return dx * dx + dy * dy + dz * dz;
}

function seededRandom(seed: number): () => number {
  let state = seed >>> 0;
  return () => {
    state = (state * 1664525 + 1013904223) >>> 0;
    return state / 4294967296;
  };
}

export function computeClusters(nodes: PointNode[], options: ClusterOptions = {}): ClusterResult {
  const cacheKey = keyForNodes(nodes, options);
  const cached = cache.get(cacheKey);
  if (cached) {
    return cached;
  }

  const points = nodes
    .filter((n) => finiteCoord(n.x) && finiteCoord(n.y) && finiteCoord(n.z))
    .map((n) => ({ id: n.id, x: n.x as number, y: n.y as number, z: n.z as number }));

  const targetClusterCount = Math.max(1, Math.min(options.targetClusterCount ?? DEFAULT_TARGET_CLUSTER_COUNT, points.length || 1));
  const maxIterations = Math.max(1, options.maxIterations ?? DEFAULT_MAX_ITERATIONS);
  const rng = seededRandom(options.seed ?? DEFAULT_SEED);

  const clusterByNodeId = new Map<string, number>();
  if (points.length === 0) {
    const emptyResult: ClusterResult = { clusterByNodeId, clusterCount: 0, cacheKey };
    cache.set(cacheKey, emptyResult);
    return emptyResult;
  }

  const centroids: Vec3[] = [];
  const used = new Set<number>();
  while (centroids.length < targetClusterCount) {
    const idx = Math.floor(rng() * points.length);
    if (used.has(idx)) {
      continue;
    }
    used.add(idx);
    centroids.push({ x: points[idx].x, y: points[idx].y, z: points[idx].z });
  }

  const assignment = new Int16Array(points.length);
  assignment.fill(-1);

  for (let iter = 0; iter < maxIterations; iter += 1) {
    let changed = false;

    for (let i = 0; i < points.length; i += 1) {
      const p = points[i];
      let bestCluster = 0;
      let bestDist = Number.POSITIVE_INFINITY;
      for (let c = 0; c < centroids.length; c += 1) {
        const d = distanceSq(p, centroids[c]);
        if (d < bestDist) {
          bestDist = d;
          bestCluster = c;
        }
      }
      if (assignment[i] !== bestCluster) {
        assignment[i] = bestCluster;
        changed = true;
      }
    }

    const sumX = new Float64Array(centroids.length);
    const sumY = new Float64Array(centroids.length);
    const sumZ = new Float64Array(centroids.length);
    const count = new Uint32Array(centroids.length);

    for (let i = 0; i < points.length; i += 1) {
      const c = assignment[i];
      if (c < 0) continue;
      sumX[c] += points[i].x;
      sumY[c] += points[i].y;
      sumZ[c] += points[i].z;
      count[c] += 1;
    }

    for (let c = 0; c < centroids.length; c += 1) {
      if (count[c] === 0) {
        const idx = Math.floor(rng() * points.length);
        centroids[c] = { x: points[idx].x, y: points[idx].y, z: points[idx].z };
      } else {
        centroids[c] = {
          x: sumX[c] / count[c],
          y: sumY[c] / count[c],
          z: sumZ[c] / count[c]
        };
      }
    }

    if (!changed) {
      break;
    }
  }

  for (let i = 0; i < points.length; i += 1) {
    clusterByNodeId.set(points[i].id, assignment[i]);
  }

  const result: ClusterResult = {
    clusterByNodeId,
    clusterCount: centroids.length,
    cacheKey
  };

  cache.set(cacheKey, result);
  if (cache.size > 16) {
    const first = cache.keys().next().value;
    if (first) {
      cache.delete(first);
    }
  }

  return result;
}
