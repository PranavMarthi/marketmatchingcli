export type RawPoint = {
  id: string;
  x?: number;
  y?: number;
  z?: number;
};

export type Bounds3D = {
  min: { x: number; y: number; z: number };
  max: { x: number; y: number; z: number };
  center: { x: number; y: number; z: number };
  span: { x: number; y: number; z: number };
};

export type PointTransformResult = {
  positions: Float32Array;
  pointCount: number;
  validPoints: Array<Required<Pick<RawPoint, "id" | "x" | "y" | "z">>>;
  bounds: Bounds3D;
  scale: number;
  suggestedCameraDistance: number;
};

const TARGET_SCENE_SPAN = 80;

function finiteNumber(value: number | undefined): value is number {
  return typeof value === "number" && Number.isFinite(value);
}

export function transformPointsForScene(points: RawPoint[]): PointTransformResult {
  const validPoints = points
    .filter((point) => finiteNumber(point.x) && finiteNumber(point.y) && finiteNumber(point.z))
    .map((point) => ({ id: point.id, x: point.x as number, y: point.y as number, z: point.z as number }));

  if (validPoints.length === 0) {
    return {
      positions: new Float32Array(),
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
  for (let i = 0; i < validPoints.length; i += 1) {
    const point = validPoints[i];
    positions[i * 3] = (point.x - centerX) * scale;
    positions[i * 3 + 1] = (point.y - centerY) * scale;
    positions[i * 3 + 2] = (point.z - centerZ) * scale;
  }

  const suggestedCameraDistance = Math.max(45, TARGET_SCENE_SPAN * 1.1);

  return {
    positions,
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
