export function createRenderSchedule(maxFramesPerSecond) {
  const intervalMs = 1000 / maxFramesPerSecond;
  let dirty = true;
  let lastFrameMs = -Infinity;
  return {
    invalidate() { dirty = true; },
    shouldRender(timeMs) {
      if (!dirty || timeMs - lastFrameMs < intervalMs) return false;
      dirty = false;
      lastFrameMs = timeMs;
      return true;
    },
  };
}
