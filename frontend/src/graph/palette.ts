export const CLUSTER_PALETTE = [
  "#8ecae6",
  "#90be6d",
  "#f9c74f",
  "#f4a261",
  "#84a59d",
  "#b8c0ff",
  "#ffafcc",
  "#ffd6a5",
  "#a0c4ff",
  "#cdb4db",
  "#95d5b2",
  "#e9c46a",
  "#f6bd60",
  "#bde0fe",
  "#c7f9cc",
  "#ffc8dd"
] as const;

export const UNKNOWN_CLUSTER_COLOR = "#d9d9d9";

export function colorForCluster(clusterId: number | undefined): string {
  if (clusterId === undefined || Number.isNaN(clusterId) || clusterId < 0) {
    return UNKNOWN_CLUSTER_COLOR;
  }
  return CLUSTER_PALETTE[clusterId % CLUSTER_PALETTE.length] ?? UNKNOWN_CLUSTER_COLOR;
}
