import { useEffect, useMemo, useRef } from "react";
import {
  Color,
  BufferGeometry,
  Float32BufferAttribute,
  InstancedMesh,
  LineBasicMaterial,
  MeshLambertMaterial,
  MeshPhysicalMaterial,
  Object3D,
  SphereGeometry,
  DoubleSide,
  Vector3,
} from "three";
import { Canvas, type ThreeEvent, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";
import { ConvexGeometry } from "three-stdlib";

import AxesGizmo from "./AxesGizmo";

type PointCloudSceneProps = {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
  edges?: Array<{
    sourceIndex: number;
    targetIndex: number;
    confidence: number;
  }>;
  neighborhoodShells?: Array<{
    id: string;
    color: [number, number, number];
    positions: Float32Array;
  }>;
  suggestedCameraDistance: number;
  pointSize: number;
  onPointClick?: (pointIndex: number) => void;
  onPointHover?: (pointIndex: number | null) => void;
};

const DAMPING_FACTOR = 0.1;
const ZOOM_SPEED = 0.55;
const ROTATE_SPEED = 0.45;
const PAN_SPEED = 0.5;
const MAX_ZOOM_OUT_CAP = 140;

const ZOOM_SIZE_ALPHA = 0.85;
const ZOOM_SIZE_MIN_SCALE = 0.08;
const ZOOM_SIZE_MAX_SCALE = 1.2;
const ZOOM_SMOOTHING = 0.14;
const SCALE_APPLY_THRESHOLD = 0.0015;
const PER_POINT_DISTANCE_MIN = 0.35;
const PER_POINT_DISTANCE_MAX = 2.2;
const PER_POINT_DISTANCE_EXPONENT = 0.45;
const SHELL_INFLATION_FACTOR = 1.02;
const SHELL_DIRECTION_COUNT = 420;

type ShellMeshData = {
  id: string;
  geometry: BufferGeometry;
  material: MeshPhysicalMaterial;
};

type EdgeMeshData = {
  geometry: BufferGeometry;
  material: LineBasicMaterial;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function buildHullPoints(positions: Float32Array): Vector3[] {
  const count = Math.floor(positions.length / 3);
  const points: Vector3[] = [];
  for (let i = 0; i < count; i += 1) {
    points.push(new Vector3(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]));
  }
  if (count < 8) return points;

  let cx = 0;
  let cy = 0;
  let cz = 0;
  for (let i = 0; i < count; i += 1) {
    cx += positions[i * 3];
    cy += positions[i * 3 + 1];
    cz += positions[i * 3 + 2];
  }
  cx /= count;
  cy /= count;
  cz /= count;

  const center = new Vector3(cx, cy, cz);
  const golden = Math.PI * (3 - Math.sqrt(5));
  const dir = new Vector3();

  for (let i = 0; i < SHELL_DIRECTION_COUNT; i += 1) {
    const y = 1 - (i / (SHELL_DIRECTION_COUNT - 1)) * 2;
    const radius = Math.sqrt(Math.max(0, 1 - y * y));
    const theta = golden * i;
    dir.set(Math.cos(theta) * radius, y, Math.sin(theta) * radius);

    let maxProj = Number.NEGATIVE_INFINITY;
    for (let j = 0; j < count; j += 1) {
      const dx = positions[j * 3] - cx;
      const dy = positions[j * 3 + 1] - cy;
      const dz = positions[j * 3 + 2] - cz;
      const proj = dx * dir.x + dy * dir.y + dz * dir.z;
      if (proj > maxProj) maxProj = proj;
    }

    if (maxProj > 0) {
      points.push(
        new Vector3(
          cx + dir.x * maxProj * 1.04,
          cy + dir.y * maxProj * 1.04,
          cz + dir.z * maxProj * 1.04
        )
      );
    }
  }

  // keep center-referenced shell smooth while ensuring all points stay enclosed
  points.push(center);
  return points;
}

function inflateGeometry(geometry: BufferGeometry, amount: number): void {
  if (amount <= 1) return;
  const attr = geometry.getAttribute("position");
  if (!attr) return;

  const center = new Vector3();
  const vertex = new Vector3();
  for (let i = 0; i < attr.count; i += 1) {
    vertex.set(attr.getX(i), attr.getY(i), attr.getZ(i));
    center.add(vertex);
  }
  center.multiplyScalar(1 / Math.max(1, attr.count));

  for (let i = 0; i < attr.count; i += 1) {
    vertex.set(attr.getX(i), attr.getY(i), attr.getZ(i));
    vertex.sub(center).multiplyScalar(amount).add(center);
    attr.setXYZ(i, vertex.x, vertex.y, vertex.z);
  }

  attr.needsUpdate = true;
  geometry.computeVertexNormals();
}

function NeighborhoodShellMeshes({
  shells,
}: {
  shells: Array<{
    id: string;
    color: [number, number, number];
    positions: Float32Array;
  }>;
}): JSX.Element {
  const shellData = useMemo<ShellMeshData[]>(() => {
    const out: ShellMeshData[] = [];
    for (const shell of shells) {
      try {
        const points = buildHullPoints(shell.positions);
        if (points.length < 4) continue;
        const geometry = new ConvexGeometry(points);
        inflateGeometry(geometry, SHELL_INFLATION_FACTOR);
        const material = new MeshPhysicalMaterial({
          color: new Color(0.34, 0.36, 0.4),
          emissive: new Color(0, 0, 0),
          transparent: true,
          opacity: 0.2,
          depthWrite: false,
          side: DoubleSide,
          metalness: 0.0,
          roughness: 0.58,
          clearcoat: 0.18,
          clearcoatRoughness: 0.5,
          transmission: 0.12,
          thickness: 0.55,
          ior: 1.12,
          reflectivity: 0.12,
        });
        out.push({ id: shell.id, geometry, material });
      } catch {
        // skip problematic hulls
      }
    }
    return out;
  }, [shells]);

  useEffect(() => {
    return () => {
      for (const shell of shellData) {
        shell.geometry.dispose();
        shell.material.dispose();
      }
    };
  }, [shellData]);

  return (
    <group>
      {shellData.map((shell) => (
        <mesh key={shell.id} geometry={shell.geometry} material={shell.material} raycast={() => {}} />
      ))}
    </group>
  );
}

function EventEdges({
  edges,
  positions,
}: {
  edges: Array<{
    sourceIndex: number;
    targetIndex: number;
    confidence: number;
  }>;
  positions: Float32Array;
}): JSX.Element | null {
  const edgeData = useMemo<EdgeMeshData | null>(() => {
    if (!edges.length) return null;

    const linePositions = new Float32Array(edges.length * 6);
    const lineColors = new Float32Array(edges.length * 6);

    for (let i = 0; i < edges.length; i += 1) {
      const edge = edges[i];
      const a = edge.sourceIndex * 3;
      const b = edge.targetIndex * 3;
      const confidence = clamp(edge.confidence, 0, 1);
      const intensity = 0.45 + confidence * 0.55;

      linePositions[i * 6] = positions[a];
      linePositions[i * 6 + 1] = positions[a + 1];
      linePositions[i * 6 + 2] = positions[a + 2];
      linePositions[i * 6 + 3] = positions[b];
      linePositions[i * 6 + 4] = positions[b + 1];
      linePositions[i * 6 + 5] = positions[b + 2];

      lineColors[i * 6] = intensity;
      lineColors[i * 6 + 1] = intensity;
      lineColors[i * 6 + 2] = intensity;
      lineColors[i * 6 + 3] = intensity;
      lineColors[i * 6 + 4] = intensity;
      lineColors[i * 6 + 5] = intensity;
    }

    const geometry = new BufferGeometry();
    geometry.setAttribute("position", new Float32BufferAttribute(linePositions, 3));
    geometry.setAttribute("color", new Float32BufferAttribute(lineColors, 3));

    const material = new LineBasicMaterial({
      vertexColors: true,
      transparent: true,
      opacity: 0.82,
      depthWrite: false,
      toneMapped: false,
    });

    return { geometry, material };
  }, [edges, positions]);

  useEffect(() => {
    return () => {
      edgeData?.geometry.dispose();
      edgeData?.material.dispose();
    };
  }, [edgeData]);

  if (!edgeData) return null;

  return <lineSegments geometry={edgeData.geometry} material={edgeData.material} raycast={() => {}} />;
}

function CloudPoints({
  positions,
  colors,
  pointSize,
  pointCount,
  suggestedCameraDistance,
  onPointClick,
  onPointHover,
}: {
  positions: Float32Array;
  colors: Float32Array;
  pointSize: number;
  pointCount: number;
  suggestedCameraDistance: number;
  onPointClick?: (pointIndex: number) => void;
  onPointHover?: (pointIndex: number | null) => void;
}): JSX.Element {
  const meshRef = useRef<InstancedMesh | null>(null);
  const dummyRef = useRef(new Object3D());
  const lastHoverIndexRef = useRef<number | null>(null);
  const geometry = useMemo(() => new SphereGeometry(1, 8, 8), []);
  const material = useMemo(
    () =>
      new MeshLambertMaterial({
        color: 0xffffff,
        transparent: false,
        opacity: 1,
      }),
    []
  );

  const smoothDistanceRef = useRef(Math.max(1, suggestedCameraDistance));
  const appliedScaleRef = useRef(1);

  const applyInstanceScale = (scaleFactor: number, cameraPosition?: Vector3) => {
    const mesh = meshRef.current;
    if (!mesh) return;
    const dummy = dummyRef.current;
    const referenceDistance = Math.max(1, suggestedCameraDistance);

    for (let i = 0; i < pointCount; i += 1) {
      const idx = i * 3;
      const px = positions[idx];
      const py = positions[idx + 1];
      const pz = positions[idx + 2];
      dummy.position.set(px, py, pz);

      let localScale = pointSize * scaleFactor;
      if (cameraPosition) {
        const dx = px - cameraPosition.x;
        const dy = py - cameraPosition.y;
        const dz = pz - cameraPosition.z;
        const distance = Math.sqrt(dx * dx + dy * dy + dz * dz);
        const distanceRatio = clamp(distance / referenceDistance, PER_POINT_DISTANCE_MIN, PER_POINT_DISTANCE_MAX);
        localScale *= Math.pow(distanceRatio, PER_POINT_DISTANCE_EXPONENT);
      }

      dummy.scale.setScalar(localScale);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);
    }

    mesh.instanceMatrix.needsUpdate = true;
  };

  useEffect(() => {
    return () => {
      geometry.dispose();
      material.dispose();
    };
  }, [geometry, material]);

  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) {
      return;
    }

    const dummy = dummyRef.current;
    const color = new Color();

    for (let i = 0; i < pointCount; i += 1) {
      const idx = i * 3;
      dummy.position.set(positions[idx], positions[idx + 1], positions[idx + 2]);
      dummy.scale.setScalar(pointSize * appliedScaleRef.current);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const r = Math.max(0.25, Math.min(1, colors[idx]));
      const g = Math.max(0.25, Math.min(1, colors[idx + 1]));
      const b = Math.max(0.25, Math.min(1, colors[idx + 2]));
      color.setRGB(r, g, b);
      mesh.setColorAt(i, color);
    }

    mesh.count = pointCount;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) {
      mesh.instanceColor.needsUpdate = true;
    }
  }, [colors, pointCount, pointSize, positions]);

  useFrame((state) => {
    const referenceDistance = Math.max(1, suggestedCameraDistance);
    const cameraDistance = Math.max(1, state.camera.position.length());
    const smoothDistance =
      smoothDistanceRef.current + (cameraDistance - smoothDistanceRef.current) * ZOOM_SMOOTHING;
    smoothDistanceRef.current = smoothDistance;

    const targetScale = clamp(
      Math.pow(smoothDistance / referenceDistance, ZOOM_SIZE_ALPHA),
      ZOOM_SIZE_MIN_SCALE,
      ZOOM_SIZE_MAX_SCALE
    );
    const prevScale = appliedScaleRef.current;
    const nextScale = prevScale + (targetScale - prevScale) * 0.2;
    if (Math.abs(nextScale - prevScale) < SCALE_APPLY_THRESHOLD) {
      return;
    }

    appliedScaleRef.current = nextScale;
    applyInstanceScale(nextScale, state.camera.position);
  });

  return (
    <instancedMesh
      ref={meshRef}
      args={[geometry, material, pointCount]}
      frustumCulled={false}
      onPointerMove={(event: ThreeEvent<PointerEvent>) => {
        event.stopPropagation();
        const nextIndex = typeof event.instanceId === "number" ? event.instanceId : null;
        if (nextIndex !== lastHoverIndexRef.current) {
          lastHoverIndexRef.current = nextIndex;
          onPointHover?.(nextIndex);
        }
      }}
      onPointerOut={() => {
        if (lastHoverIndexRef.current !== null) {
          lastHoverIndexRef.current = null;
          onPointHover?.(null);
        }
      }}
      onClick={(event: ThreeEvent<MouseEvent>) => {
        event.stopPropagation();
        if (typeof event.instanceId === "number") {
          onPointClick?.(event.instanceId);
        }
      }}
    />
  );
}

export default function PointCloudScene({
  positions,
  colors,
  pointCount,
  edges = [],
  neighborhoodShells = [],
  suggestedCameraDistance,
  pointSize,
  onPointClick,
  onPointHover,
}: PointCloudSceneProps): JSX.Element {
  const minDistance = Math.max(4, suggestedCameraDistance * 0.08);
  const maxDistance = Math.min(MAX_ZOOM_OUT_CAP, Math.max(40, suggestedCameraDistance * 1.6));

  return (
    <Canvas
      camera={{
        fov: 58,
        near: 0.1,
        far: 5000,
        position: [0, 0, suggestedCameraDistance]
      }}
      dpr={[1, Math.min(window.devicePixelRatio || 1, 2)]}
      gl={{ antialias: true, alpha: true, powerPreference: "high-performance" }}
      shadows={false}
    >
      <color attach="background" args={["#05060a"]} />
      <ambientLight intensity={0.3} />
      <hemisphereLight args={["#86a8ff", "#0a0f1a", 0.42]} />
      <directionalLight position={[28, 40, 36]} intensity={0.55} color="#d7e4ff" />
      <directionalLight position={[-22, -16, -28]} intensity={0.2} color="#79d8ff" />
      <pointLight position={[18, 16, 20]} intensity={0.35} color="#dbe4ff" />

      <AxesGizmo size={38} showGrid />
      {neighborhoodShells.length > 0 && <NeighborhoodShellMeshes shells={neighborhoodShells} />}
      {edges.length > 0 && <EventEdges edges={edges} positions={positions} />}
      {pointCount > 0 && (
        <CloudPoints
          positions={positions}
          colors={colors}
          pointSize={pointSize}
          pointCount={pointCount}
          suggestedCameraDistance={suggestedCameraDistance}
          onPointClick={onPointClick}
          onPointHover={onPointHover}
        />
      )}

      <OrbitControls
        makeDefault
        enableDamping
        dampingFactor={DAMPING_FACTOR}
        zoomSpeed={ZOOM_SPEED}
        rotateSpeed={ROTATE_SPEED}
        panSpeed={PAN_SPEED}
        minDistance={minDistance}
        maxDistance={maxDistance}
      />
    </Canvas>
  );
}
