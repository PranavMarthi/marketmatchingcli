export type RawPoint = {
  id: string;
  x?: number;
  y?: number;
  z?: number;
  cluster_id?: string | null;
};

export type Bounds3D = {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
  center: { x: number; y: number; z: number };
  span: { x: number; y: number; z: number };
};

export type PointTransformResult = {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
  validPoints: Array<Required<Pick<RawPoint, "id" | "x" | "y" | "z">> & { cluster_id?: string | null }>;
  bounds: Bounds3D;
  scale: number;
  suggestedCameraDistance: number;
};

const TARGET_SCENE_SPAN = 80;
const TOP_CLUSTER_COLOR_COUNT = 24;
const MIN_CLUSTER_POINTS_FOR_COLOR = 20;

// k-approximate-nearest-neighbor clustering knobs.
// Adjust MAX_CLUSTER_NEIGHBOR_DISTANCE to make clusters looser/tighter.
const APPROX_K_NEIGHBORS = 12;
const MAX_CLUSTER_NEIGHBOR_DISTANCE = 1.8;

const FALLBACK_UNCLUSTERED: readonly [number, number, number] = [0.82, 0.86, 0.94];

function finiteNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

class UnionFind {
  private readonly parent: Int32Array;
  private readonly rank: Int8Array;

  constructor(size: number) {
    this.parent = new Int32Array(size);
    this.rank = new Int8Array(size);
    for (let i = 0; i < size; i += 1) {
      this.parent[i] = i;
      this.rank[i] = 0;
    }
  }

  find(x: number): number {
    let p = this.parent[x];
    while (p !== this.parent[p]) {
      p = this.parent[p];
    }
    let cur = x;
    while (cur !== p) {
      const next = this.parent[cur];
      this.parent[cur] = p;
      cur = next;
    }
    return p;
  }

  union(a: number, b: number): void {
    const ra = this.find(a);
    const rb = this.find(b);
    if (ra === rb) return;

    if (this.rank[ra] < this.rank[rb]) {
      this.parent[ra] = rb;
    } else if (this.rank[ra] > this.rank[rb]) {
      this.parent[rb] = ra;
    } else {
      this.parent[rb] = ra;
      this.rank[ra] += 1;
    }
  }
}

function hashString(value: string): number {
  let hash = 2166136261;
  for (let i = 0; i < value.length; i += 1) {
    hash ^= value.charCodeAt(i);
    hash = Math.imul(hash, 16777619);
  }
  return hash >>> 0;
}

function hslToRgb(h: number, s: number, l: number): readonly [number, number, number] {
  const hue2rgb = (p: number, q: number, t: number): number => {
    let x = t;
    if (x < 0) x += 1;
    if (x > 1) x -= 1;
    if (x < 1 / 6) return p + (q - p) * 6 * x;
    if (x < 1 / 2) return q;
    if (x < 2 / 3) return p + (q - p) * (2 / 3 - x) * 6;
    return p;
  };

  if (s === 0) {
    return [l, l, l];
  }

  const q = l < 0.5 ? l * (1 + s) : l + s - l * s;
  const p = 2 * l - q;

  return [
    hue2rgb(p, q, h + 1 / 3),
    hue2rgb(p, q, h),
    hue2rgb(p, q, h - 1 / 3),
  ];
}

function colorForTopCluster(clusterId: string): readonly [number, number, number] {
  const hash = hashString(clusterId);
  const hue = (hash % 360) / 360;
  const sat = 0.88;
  const light = 0.56;
  return hslToRgb(hue, sat, light);
}

function colorForSmallCluster(clusterId: string): readonly [number, number, number] {
  const hash = hashString(clusterId);
  const hue = (hash % 360) / 360;
  const sat = 0.42;
  const light = 0.62;
  return hslToRgb(hue, sat, light);
}

function buildClusterCounts(labels: string[]): Map<string, number> {
  const counts = new Map<string, number>();
  for (const clusterId of labels) {
    counts.set(clusterId, (counts.get(clusterId) ?? 0) + 1);
  }
  return counts;
}

function buildTopClusterSet(clusterCounts: Map<string, number>): Set<string> {
  const ranked = [...clusterCounts.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, TOP_CLUSTER_COLOR_COUNT)
    .map(([clusterId]) => clusterId);
  return new Set(ranked);
}

type ScaledPoint = {
  x: number;
  y: number;
  z: number;
};

function cellKey(ix: number, iy: number, iz: number): string {
  return `${ix}:${iy}:${iz}`;
}

function computeApproxKnnClusterLabels(points: ScaledPoint[]): string[] {
  const count = points.length;
  if (count === 0) return [];

  const uf = new UnionFind(count);
  const cellSize = MAX_CLUSTER_NEIGHBOR_DISTANCE;
  const cellMap = new Map<string, number[]>();

  for (let i = 0; i < count; i += 1) {
    const p = points[i];
    const cx = Math.floor(p.x / cellSize);
    const cy = Math.floor(p.y / cellSize);
    const cz = Math.floor(p.z / cellSize);
    const key = cellKey(cx, cy, cz);
    const bucket = cellMap.get(key);
    if (bucket) {
      bucket.push(i);
    } else {
      cellMap.set(key, [i]);
    }
  }

  const maxDistSq = MAX_CLUSTER_NEIGHBOR_DISTANCE * MAX_CLUSTER_NEIGHBOR_DISTANCE;
  for (let i = 0; i < count; i += 1) {
    const p = points[i];
    const cx = Math.floor(p.x / cellSize);
    const cy = Math.floor(p.y / cellSize);
    const cz = Math.floor(p.z / cellSize);

    const best: Array<{ idx: number; d2: number }> = [];
    for (let dx = -1; dx <= 1; dx += 1) {
      for (let dy = -1; dy <= 1; dy += 1) {
        for (let dz = -1; dz <= 1; dz += 1) {
          const bucket = cellMap.get(cellKey(cx + dx, cy + dy, cz + dz));
          if (!bucket) continue;

          for (const j of bucket) {
            if (j <= i) continue;
            const q = points[j];
            const ddx = p.x - q.x;
            const ddy = p.y - q.y;
            const ddz = p.z - q.z;
            const d2 = ddx * ddx + ddy * ddy + ddz * ddz;
            if (d2 > maxDistSq) continue;

            if (best.length < APPROX_K_NEIGHBORS) {
              best.push({ idx: j, d2 });
              best.sort((a, b) => a.d2 - b.d2);
            } else if (d2 < best[best.length - 1].d2) {
              best[best.length - 1] = { idx: j, d2 };
              best.sort((a, b) => a.d2 - b.d2);
            }
          }
        }
      }
    }

    for (const n of best) {
      uf.union(i, n.idx);
    }
  }

  const rootToLabel = new Map<number, string>();
  const labels = new Array<string>(count);
  let next = 0;
  for (let i = 0; i < count; i += 1) {
    const root = uf.find(i);
    let label = rootToLabel.get(root);
    if (!label) {
      label = String(next);
      rootToLabel.set(root, label);
      next += 1;
    }
    labels[i] = label;
  }
  return labels;
}

export function transformPointsForScene(points: RawPoint[]): PointTransformResult {
  const validPoints = points
    .filter((point) => finiteNumber(point.x) && finiteNumber(point.y) && finiteNumber(point.z))
    .map((point) => ({
      id: point.id,
      x: point.x as number,
      y: point.y as number,
      z: point.z as number,
      cluster_id: point.cluster_id ?? null,
    }));

  if (validPoints.length === 0) {
    return {
      positions: new Float32Array(),
      colors: new Float32Array(),
      pointCount: 0,
      validPoints,
      bounds: {
        min: { x: 0, y: 0, z: 0 },
        max: { x: 0, y: 0, z: 0 },
        center: { x: 0, y: 0, z: 0 },
        span: { x: 0, y: 0, z: 0 }
      },
      scale: 1,
      suggestedCameraDistance: 60
    };
  }

  let minX = Number.POSITIVE_INFINITY;
  let minY = Number.POSITIVE_INFINITY;
  let minZ = Number.POSITIVE_INFINITY;
  let maxX = Number.NEGATIVE_INFINITY;
  let maxY = Number.NEGATIVE_INFINITY;
  let maxZ = Number.NEGATIVE_INFINITY;

  for (const point of validPoints) {
    if (point.x < minX) minX = point.x;
    if (point.y < minY) minY = point.y;
    if (point.z < minZ) minZ = point.z;
    if (point.x > maxX) maxX = point.x;
    if (point.y > maxY) maxY = point.y;
    if (point.z > maxZ) maxZ = point.z;
  }

  const centerX = (minX + maxX) / 2;
  const centerY = (minY + maxY) / 2;
  const centerZ = (minZ + maxZ) / 2;

  const spanX = Math.max(1e-6, maxX - minX);
  const spanY = Math.max(1e-6, maxY - minY);
  const spanZ = Math.max(1e-6, maxZ - minZ);
  const maxSpan = Math.max(spanX, spanY, spanZ);

  const scale = TARGET_SCENE_SPAN / maxSpan;
  const positions = new Float32Array(validPoints.length * 3);
  const scaledPoints = new Array<ScaledPoint>(validPoints.length);
  for (let i = 0; i < validPoints.length; i += 1) {
    const point = validPoints[i];
    const sx = (point.x - centerX) * scale;
    const sy = (point.y - centerY) * scale;
    const sz = (point.z - centerZ) * scale;
    positions[i * 3] = sx;
    positions[i * 3 + 1] = sy;
    positions[i * 3 + 2] = sz;
    scaledPoints[i] = { x: sx, y: sy, z: sz };
  }

  const clusterLabels = computeApproxKnnClusterLabels(scaledPoints);

  const colors = new Float32Array(validPoints.length * 3);
  const clusterCounts = buildClusterCounts(clusterLabels);
  const topClusters = buildTopClusterSet(clusterCounts);

  for (let i = 0; i < validPoints.length; i += 1) {
    const clusterId = clusterLabels[i];
    let rgb = FALLBACK_UNCLUSTERED;
    const clusterSize = clusterCounts.get(clusterId) ?? 0;
    if (clusterSize < MIN_CLUSTER_POINTS_FOR_COLOR) {
      rgb = [1.0, 1.0, 1.0];
    } else {
      rgb = topClusters.has(clusterId)
        ? colorForTopCluster(clusterId)
        : colorForSmallCluster(clusterId);
    }
    colors[i * 3] = rgb[0];
    colors[i * 3 + 1] = rgb[1];
    colors[i * 3 + 2] = rgb[2];
  }

  const suggestedCameraDistance = Math.max(45, TARGET_SCENE_SPAN * 1.1);

  return {
    positions,
    colors,
    pointCount: validPoints.length,
    validPoints,
    bounds: {
      min: { x: minX, y: minY, z: minZ },
      max: { x: maxX, y: maxY, z: maxZ },
      center: { x: centerX, y: centerY, z: centerZ },
      span: { x: spanX, y: spanY, z: spanZ }
    },
    scale,
    suggestedCameraDistance
  };
}
