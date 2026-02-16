import { useEffect, useMemo, useRef } from "react";
import {
  Color,
  InstancedMesh,
  MeshLambertMaterial,
  Object3D,
  SphereGeometry,
  Vector3,
} from "three";
import { Canvas, type ThreeEvent, useFrame } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import AxesGizmo from "./AxesGizmo";

type PointCloudSceneProps = {
  positions: Float32Array;
  colors: Float32Array;
  pointCount: number;
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

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
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
      <ambientLight intensity={0.45} />
      <pointLight position={[30, 30, 40]} intensity={0.55} />

      <AxesGizmo size={38} showGrid />
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
