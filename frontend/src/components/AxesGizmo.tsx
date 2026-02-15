import { Line } from "@react-three/drei";

type AxesGizmoProps = {
  size?: number;
  showGrid?: boolean;
};

export default function AxesGizmo({ size = 40, showGrid = true }: AxesGizmoProps): JSX.Element {
  return (
    <group>
      {showGrid && <gridHelper args={[size * 2, 24, "#1b2430", "#0f141c"]} />}
      <Line points={[[-size, 0, 0], [size, 0, 0]]} color="#ff7b7b" transparent opacity={0.58} />
      <Line points={[[0, -size, 0], [0, size, 0]]} color="#8dffab" transparent opacity={0.58} />
      <Line points={[[0, 0, -size], [0, 0, size]]} color="#81a8ff" transparent opacity={0.58} />
    </group>
  );
}
