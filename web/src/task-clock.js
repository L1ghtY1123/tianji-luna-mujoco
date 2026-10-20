export function createTaskClock({ now = () => performance.now(), wallNow = () => Date.now() } = {}) {
  function emptyState() {
    return { runId: null, seconds: 0, startedAt: null, calibrated: false };
  }

  let state = emptyState();
  let pending = null;

  function elapsedSeconds() {
    return state.seconds + (state.startedAt === null ? 0 : Math.max(0, now() - state.startedAt) / 1000);
  }

  function sync(run, events = [], serverTime) {
    if (!run?.status) return;
    const confirmingStart = pending?.action === 'start';
    if (pending?.action === 'reset' && run.status !== 'idle') return;
    if (confirmingStart && (run.status === 'idle' || (run.run_id && run.run_id === pending.previous.runId))) return;
    pending = null;

    if (run.status === 'idle') {
      state = emptyState();
      return;
    }

    if (run.run_id !== state.runId) {
      // Keep click-to-request progress until this new run supplies its start event.
      if (!confirmingStart) state = emptyState();
      state.runId = run.run_id;
    }

    if (run.status === 'running') {
      const start = events.find(event => event.kind === 'run_start' && Number.isFinite(event.time));
      if (!state.calibrated && start) {
        // Server timestamps avoid client clock skew; older servers use the browser wall clock.
        const currentTime = Number.isFinite(serverTime) ? serverTime : wallNow() / 1000;
        state.seconds = Math.max(0, currentTime - start.time);
        state.startedAt = now();
        state.calibrated = true;
      }
      if (state.startedAt === null) state.startedAt = now();
      return;
    }

    const duration = run.metrics?.run_duration_s;
    state.seconds = Number.isFinite(duration) ? Math.max(0, duration) : elapsedSeconds();
    state.startedAt = null;
  }

  return {
    onStart() {
      pending = { action: 'start', previous: { ...state } };
      state = emptyState();
      state.startedAt = now();
    },
    cancelPending() {
      if (!pending) return;
      state = pending.previous;
      pending = null;
    },
    reset() {
      pending = { action: 'reset', previous: { ...state } };
      state = emptyState();
    },
    sync,
    elapsedSeconds,
    read: elapsedSeconds,
  };
}
