export type RawPoint = {
  id: string;
  x?: number;
  y?: number;
  z?: number;
  cluster_id?: string | null;
  neighborhood_key?: string | null;
  neighborhood_label?: string | null;
  local_cluster_id?: number | null;
  local_distortion?: number | null;
  stitch_distortion?: number | null;
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
  validPoints: Array<
    Required<Pick<RawPoint, "id" | "x" | "y" | "z">> & {
      cluster_id?: string | null;
      neighborhood_key?: string | null;
      neighborhood_label?: string | null;
      local_cluster_id?: number | null;
      local_distortion?: number | null;
      stitch_distortion?: number | null;
    }
  >;
  bounds: Bounds3D;
  scale: number;
  suggestedCameraDistance: number;
};

export type PointTransformOptions = {
  colorBy?: "neighborhood" | "local_cluster";
  intraClusterScale?: number;
  macroSeparation?: number;
};

const TARGET_SCENE_SPAN = 80;
const TOP_CLUSTER_COLOR_COUNT = 24;
const MIN_CLUSTER_POINTS_FOR_COLOR = 20;

// k-approximate-nearest-neighbor clustering knobs.
// Adjust MAX_CLUSTER_NEIGHBOR_DISTANCE to make clusters looser/tighter.
const APPROX_K_NEIGHBORS = 12;
const MAX_CLUSTER_NEIGHBOR_DISTANCE = 1.8;
const CLUSTER_ID_COVERAGE_THRESHOLD = 0.6;

const FALLBACK_UNCLUSTERED: readonly [number, number, number] = [0.82, 0.86, 0.94];

const MACRO_HUE: Record<string, number> = {
  sports: 145, // green-cyan
  politics: 355, // red
  crypto: 220, // blue
  geopolitics: 32, // orange
  economy: 52, // amber
  misc: 270, // violet
  tech: 198, // cyan-blue
  pop_culture: 314, // magenta
  science: 176, // teal
  weather: 86, // lime
};

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
  const sat = 0.82;
  const light = 0.56;
  return hslToRgb(hue, sat, light);
}

function colorForSmallCluster(clusterId: string): readonly [number, number, number] {
  const hash = hashString(clusterId);
  const hue = (hash % 360) / 360;
  const sat = 0.52;
  const light = 0.62;
  return hslToRgb(hue, sat, light);
}

function extractMacroKey(clusterLabel: string): string {
  const parts = clusterLabel.split("::");
  return parts[0] || "misc";
}

function baseHueForMacro(macroKey: string): number {
  const normalized = macroKey.trim().toLowerCase();
  if (normalized in MACRO_HUE) return MACRO_HUE[normalized];
  return hashString(normalized) % 360;
}

function colorForNeighborhood(clusterLabel: string): readonly [number, number, number] {
  const hue = baseHueForMacro(extractMacroKey(clusterLabel)) / 360;
  return hslToRgb(hue, 0.82, 0.56);
}

function colorForLocalClusterLabel(clusterLabel: string): readonly [number, number, number] {
  const macro = extractMacroKey(clusterLabel);
  const baseHue = baseHueForMacro(macro);
  const match = /::c(\d+)$/.exec(clusterLabel);
  const clusterIndex = match ? Number.parseInt(match[1], 10) : 0;

  const hueOffset = ((clusterIndex * 37) % 72) - 36; // stay in macro family but distinguish
  const hue = ((baseHue + hueOffset + 360) % 360) / 360;
  const sat = 0.68 + ((clusterIndex * 17) % 7) * 0.03;
  const light = 0.46 + ((clusterIndex * 13) % 7) * 0.04;
  return hslToRgb(hue, Math.min(0.92, sat), Math.min(0.72, light));
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

function shouldUseBackendClusterIds(points: Array<{ cluster_id?: string | null }>): boolean {
  if (points.length === 0) return false;
  let assigned = 0;
  for (const point of points) {
    if (point.cluster_id && point.cluster_id.length > 0) {
      assigned += 1;
    }
  }
  return assigned / points.length >= CLUSTER_ID_COVERAGE_THRESHOLD;
}

function buildClusterLabelsFromBackend(points: Array<{ cluster_id?: string | null }>): string[] {
  return points.map((point) => (point.cluster_id && point.cluster_id.length > 0 ? point.cluster_id : "-1"));
}

type ScaledPoint = {
  x: number;
  y: number;
  z: number;
};

function expandIntraClusterSpacing<T extends { x: number; y: number; z: number; neighborhood_key?: string | null }>(
  points: T[],
  factor: number
): T[] {
  const safeFactor = Number.isFinite(factor) ? Math.min(2.2, Math.max(1.0, factor)) : 1.0;
  if (safeFactor <= 1.0001) return points;

  const groups = new Map<string, number[]>();
  for (let i = 0; i < points.length; i += 1) {
    const key = points[i].neighborhood_key ?? "misc";
    const arr = groups.get(key);
    if (arr) arr.push(i);
    else groups.set(key, [i]);
  }

  const out = points.map((p) => ({ ...p }));
  for (const idxs of groups.values()) {
    if (idxs.length < 3) continue;
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const idx of idxs) {
      cx += points[idx].x;
      cy += points[idx].y;
      cz += points[idx].z;
    }
    cx /= idxs.length;
    cy /= idxs.length;
    cz /= idxs.length;

    for (const idx of idxs) {
      const dx = points[idx].x - cx;
      const dy = points[idx].y - cy;
      const dz = points[idx].z - cz;
      out[idx].x = cx + dx * safeFactor;
      out[idx].y = cy + dy * safeFactor;
      out[idx].z = cz + dz * safeFactor;
    }
  }
  return out;
}

function expandInterClusterSpacing<T extends { x: number; y: number; z: number; neighborhood_key?: string | null }>(
  points: T[],
  factor: number
): T[] {
  const safeFactor = Number.isFinite(factor) ? Math.min(4.0, Math.max(0.55, factor)) : 1.0;
  if (Math.abs(safeFactor - 1.0) < 0.0001) return points;

  const groups = new Map<string, number[]>();
  for (let i = 0; i < points.length; i += 1) {
    const key = points[i].neighborhood_key ?? "misc";
    const arr = groups.get(key);
    if (arr) arr.push(i);
    else groups.set(key, [i]);
  }

  const out = points.map((p) => ({ ...p }));

  let gx = 0;
  let gy = 0;
  let gz = 0;
  for (const p of points) {
    gx += p.x;
    gy += p.y;
    gz += p.z;
  }
  gx /= points.length;
  gy /= points.length;
  gz /= points.length;

  for (const idxs of groups.values()) {
    if (idxs.length === 0) continue;
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const idx of idxs) {
      cx += points[idx].x;
      cy += points[idx].y;
      cz += points[idx].z;
    }
    cx /= idxs.length;
    cy /= idxs.length;
    cz /= idxs.length;

    const separationBoost = (safeFactor - 1) * 1.35;
    const shiftX = (cx - gx) * separationBoost;
    const shiftY = (cy - gy) * separationBoost;
    const shiftZ = (cz - gz) * separationBoost;

    for (const idx of idxs) {
      out[idx].x = points[idx].x + shiftX;
      out[idx].y = points[idx].y + shiftY;
      out[idx].z = points[idx].z + shiftZ;
    }
  }

  // Enforce a balanced macro-radius band so no single neighborhood drifts too far
  // while still keeping visible separation between all neighborhoods.
  const groupCentroids = new Map<string, { x: number; y: number; z: number; idxs: number[] }>();
  for (const [key, idxs] of groups.entries()) {
    if (idxs.length === 0) continue;
    let cx = 0;
    let cy = 0;
    let cz = 0;
    for (const idx of idxs) {
      cx += out[idx].x;
      cy += out[idx].y;
      cz += out[idx].z;
    }
    groupCentroids.set(key, {
      x: cx / idxs.length,
      y: cy / idxs.length,
      z: cz / idxs.length,
      idxs,
    });
  }

  const centroidRadii: number[] = [];
  for (const c of groupCentroids.values()) {
    const dx = c.x - gx;
    const dy = c.y - gy;
    const dz = c.z - gz;
    centroidRadii.push(Math.sqrt(dx * dx + dy * dy + dz * dz));
  }

  if (centroidRadii.length >= 3) {
    const sorted = [...centroidRadii].sort((a, b) => a - b);
    const q = (p: number): number => {
      const idx = Math.max(0, Math.min(sorted.length - 1, Math.floor((sorted.length - 1) * p)));
      return sorted[idx];
    };

    const median = q(0.5);
    const minAllowed = Math.max(median * 0.82, q(0.3));
    const maxAllowed = Math.max(minAllowed + 1e-6, Math.min(median * 1.16, q(0.78)));

    for (const c of groupCentroids.values()) {
      const vx = c.x - gx;
      const vy = c.y - gy;
      const vz = c.z - gz;
      const radius = Math.sqrt(vx * vx + vy * vy + vz * vz);
      if (!Number.isFinite(radius) || radius < 1e-8) continue;

      let targetRadius = radius;
      if (radius > maxAllowed) {
        targetRadius = maxAllowed;
      } else if (radius < minAllowed) {
        targetRadius = minAllowed;
      }
      const radialScale = targetRadius / radius;
      if (Math.abs(radialScale - 1) < 1e-3) continue;

      const targetCx = gx + vx * radialScale;
      const targetCy = gy + vy * radialScale;
      const targetCz = gz + vz * radialScale;
      const dxShift = targetCx - c.x;
      const dyShift = targetCy - c.y;
      const dzShift = targetCz - c.z;

      for (const idx of c.idxs) {
        out[idx].x += dxShift;
        out[idx].y += dyShift;
        out[idx].z += dzShift;
      }
    }

    // Pairwise centroid separation: enforce clear "island" separation
    // while keeping everything inside a compact spherical shell.
    const recomputeCentroids = (): Array<{ key: string; x: number; y: number; z: number; idxs: number[] }> => {
      const list: Array<{ key: string; x: number; y: number; z: number; idxs: number[] }> = [];
      for (const [key, idxs] of groups.entries()) {
        if (idxs.length === 0) continue;
        let cx = 0;
        let cy = 0;
        let cz = 0;
        for (const idx of idxs) {
          cx += out[idx].x;
          cy += out[idx].y;
          cz += out[idx].z;
        }
        list.push({ key, x: cx / idxs.length, y: cy / idxs.length, z: cz / idxs.length, idxs });
      }
      return list;
    };

    const centroids0 = recomputeCentroids();
    if (centroids0.length >= 3) {
      const shellRadius = median;
      const desiredMin = shellRadius * 1.05;

      for (let iter = 0; iter < 10; iter += 1) {
        const centroids = recomputeCentroids();

        for (let i = 0; i < centroids.length; i += 1) {
          for (let j = i + 1; j < centroids.length; j += 1) {
            const a = centroids[i];
            const b = centroids[j];
            const dx = b.x - a.x;
            const dy = b.y - a.y;
            const dz = b.z - a.z;
            const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);
            if (!Number.isFinite(dist) || dist < 1e-6 || dist >= desiredMin) continue;

            const push = (desiredMin - dist) * 0.24;
            const ux = dx / dist;
            const uy = dy / dist;
            const uz = dz / dist;

            for (const idx of a.idxs) {
              out[idx].x -= ux * push;
              out[idx].y -= uy * push;
              out[idx].z -= uz * push;
            }
            for (const idx of b.idxs) {
              out[idx].x += ux * push;
              out[idx].y += uy * push;
              out[idx].z += uz * push;
            }
          }
        }

        // Keep centroids in a compact spherical domain after each repel pass.
        const after = recomputeCentroids();
        for (const c of after) {
          const vx = c.x - gx;
          const vy = c.y - gy;
          const vz = c.z - gz;
          const r = Math.sqrt(vx * vx + vy * vy + vz * vz);
          if (!Number.isFinite(r) || r < 1e-8) continue;

          let targetR = r;
          if (r < minAllowed) targetR = minAllowed;
          if (r > maxAllowed) targetR = maxAllowed;
          const scale = targetR / r;
          if (Math.abs(scale - 1) < 1e-3) continue;

          const tx = gx + vx * scale;
          const ty = gy + vy * scale;
          const tz = gz + vz * scale;
          const sx = tx - c.x;
          const sy = ty - c.y;
          const sz = tz - c.z;

          for (const idx of c.idxs) {
            out[idx].x += sx;
            out[idx].y += sy;
            out[idx].z += sz;
          }
        }
      }
    }
  }

  return out;
}

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

export function transformPointsForScene(
  points: RawPoint[],
  options: PointTransformOptions = {}
): PointTransformResult {
  const basePoints = points
    .filter((point) => finiteNumber(point.x) && finiteNumber(point.y) && finiteNumber(point.z))
    .map((point) => ({
      id: point.id,
      x: point.x as number,
      y: point.y as number,
      z: point.z as number,
      cluster_id: point.cluster_id ?? null,
      neighborhood_key: point.neighborhood_key ?? null,
      neighborhood_label: point.neighborhood_label ?? null,
      local_cluster_id: point.local_cluster_id ?? null,
      local_distortion: point.local_distortion ?? null,
      stitch_distortion: point.stitch_distortion ?? null,
    }));

  const intraClusterScale = options.intraClusterScale ?? 1.0;
  const macroSeparation = options.macroSeparation ?? 1.0;
  const withMacroSpacing = expandInterClusterSpacing(basePoints, macroSeparation);
  const validPoints = expandIntraClusterSpacing(withMacroSpacing, intraClusterScale);

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

  const colorBy = options.colorBy ?? "neighborhood";

  let clusterLabels: string[];
  if (colorBy === "local_cluster") {
    clusterLabels = validPoints.map((p) => {
      if (p.local_cluster_id !== null && p.local_cluster_id !== undefined) {
        const key = p.neighborhood_key ?? "misc::unknown";
        return `${key}::c${String(p.local_cluster_id)}`;
      }
      return p.neighborhood_key ?? p.cluster_id ?? "-1";
    });
  } else {
    const hasNeighborhood = validPoints.some((p) => p.neighborhood_key && p.neighborhood_key.length > 0);
    if (hasNeighborhood) {
      clusterLabels = validPoints.map((p) => p.neighborhood_key ?? p.cluster_id ?? "-1");
    } else {
      clusterLabels = shouldUseBackendClusterIds(validPoints)
        ? buildClusterLabelsFromBackend(validPoints)
        : computeApproxKnnClusterLabels(scaledPoints);
    }
  }

  const colors = new Float32Array(validPoints.length * 3);
  const clusterCounts = buildClusterCounts(clusterLabels);
  const topClusters = buildTopClusterSet(clusterCounts);

  for (let i = 0; i < validPoints.length; i += 1) {
    const clusterId = clusterLabels[i];
    let rgb = FALLBACK_UNCLUSTERED;
    const clusterSize = clusterCounts.get(clusterId) ?? 0;
    if (colorBy === "neighborhood") {
      rgb = colorForNeighborhood(clusterId);
      if (clusterSize < MIN_CLUSTER_POINTS_FOR_COLOR) {
        // keep small groups in macro family, just dimmer.
        rgb = hslToRgb(baseHueForMacro(extractMacroKey(clusterId)) / 360, 0.35, 0.62);
      }
    } else if (colorBy === "local_cluster") {
      rgb = colorForLocalClusterLabel(clusterId);
      if (clusterSize < MIN_CLUSTER_POINTS_FOR_COLOR) {
        const macroHue = baseHueForMacro(extractMacroKey(clusterId)) / 360;
        rgb = hslToRgb(macroHue, 0.45, 0.58);
      }
    } else {
      if (clusterSize < MIN_CLUSTER_POINTS_FOR_COLOR) {
        rgb = colorForSmallCluster(clusterId);
      } else {
        rgb = topClusters.has(clusterId) ? colorForTopCluster(clusterId) : colorForSmallCluster(clusterId);
      }
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
