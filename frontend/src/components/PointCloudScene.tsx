import { useEffect, useMemo, useRef } from "react";
import { AdditiveBlending, BufferAttribute, BufferGeometry, Color } from "three";
import { Canvas } from "@react-three/fiber";
import { OrbitControls } from "@react-three/drei";

import AxesGizmo from "./AxesGizmo";

type PointCloudSceneProps = {
  positions: Float32Array;
  pointCount: number;
  suggestedCameraDistance: number;
  pointSize: number;
};

const DAMPING_FACTOR = 0.1;
const ZOOM_SPEED = 0.55;
const ROTATE_SPEED = 0.45;
const PAN_SPEED = 0.5;
const MAX_ZOOM_OUT_CAP = 140;

function CloudPoints({ positions, pointSize }: { positions: Float32Array; pointSize: number }): JSX.Element {
  const geometryRef = useRef<BufferGeometry | null>(null);

  const geometry = useMemo(() => {
    const g = new BufferGeometry();
    g.setAttribute("position", new BufferAttribute(positions, 3));
    g.computeBoundingSphere();
    return g;
  }, [positions]);

  useEffect(() => {
    geometryRef.current = geometry;
    return () => {
      geometry.dispose();
      geometryRef.current = null;
    };
  }, [geometry]);

  return (
    <points geometry={geometry} frustumCulled>
      <pointsMaterial
        color={new Color("#dce5ff")}
        size={pointSize}
        transparent
        opacity={0.85}
        depthWrite={false}
        sizeAttenuation
        blending={AdditiveBlending}
      />
    </points>
  );
}

export default function PointCloudScene({
  positions,
  pointCount,
  suggestedCameraDistance,
  pointSize
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
      {pointCount > 0 && <CloudPoints positions={positions} pointSize={pointSize} />}

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
