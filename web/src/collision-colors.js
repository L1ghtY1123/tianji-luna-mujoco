// Stable per-view identity colors; an excluded layer keeps its identity color.
export function createCollisionColors() {
  const colors = new Map();
  const palette = ['#62b5e5', '#e7af65', '#b29ade', '#70c6ab', '#e88da8', '#d4cf83', '#7ccfd4'];
  return layer => {
    if (layer.role === 'measured_table') return '#91a6b6';
    const id = layer.instance_id ?? layer.name;
    if (!colors.has(id)) colors.set(id, palette[colors.size] ?? `hsl(${(colors.size * 137.508) % 360} 70% 65%)`);
    return colors.get(id);
  };
}
