import { isTaskLedger, renderTaskObjects } from './task-ledger.js';
import { renderPolicyGraph } from './replay-policy.js';
import { fitGraphCanvas } from './replay-graph.js';

export function labelsForFrame(labels, frameId, generation, cameraId = '3') {
  return labels?.observation_id === frameId && labels.generation === generation && Array.isArray(labels.items)
    ? labels.items.filter(item => (item.camera_id ?? '3') === cameraId && typeof item.id === 'string'
      && Number.isFinite(item.x) && Number.isFinite(item.y)) : [];
}

export function createObjectList(container, {document: ownerDocument = globalThis.document} = {}) {
  let graphDetails = null;
  return {render(progress, bindings = null, policyGraph = null, executionEvents = [], runId = null) {
    const graphOpen = graphDetails?.open ?? false;
    const scopeId = graphDetails?.querySelector('.graph-scope:not([hidden])')?.dataset.scopeId;
    const previousCanvas = graphDetails?.querySelector('.graph-scope:not([hidden]) .policy-canvas');
    const scale = previousCanvas?.dataset.scale;
    const scroll = [previousCanvas?.scrollLeft ?? 0, previousCanvas?.scrollTop ?? 0];
    graphDetails = null;
    if (isTaskLedger(progress)) {
      container.replaceChildren();
      container.hidden = false;
      renderTaskObjects(container, progress, bindings);
      if (policyGraph) {
        const details = ownerDocument.createElement('details');
        details.open = graphOpen;
        graphDetails = details;
        const title = ownerDocument.createElement('summary');
        title.textContent = policyGraph.id === 'terminal_failure_return' ? '失败回位收尾图' : '完整任务图（含失败收尾）';
        const tree = ownerDocument.createElement('div');
        tree.className = 'policy-tree';
        const versionStart = executionEvents.findLastIndex(event => ['skill_program', 'policy_revised'].includes(event.kind));
        const scopeExists = scopeId === policyGraph.id || policyGraph.subgraphs?.some(scope => scope.id === scopeId);
        renderPolicyGraph(tree, policyGraph, executionEvents.slice(versionStart + 1), {},
          {scopeId: scopeExists ? scopeId : policyGraph.id});
        details.append(title);
        if (runId) {
          const link = ownerDocument.createElement('a');
          link.className = 'button'; link.textContent = '全宽查看任务图 ↗';
          link.href = `/#replay?run=${encodeURIComponent(runId)}&view=graph&standalone=1`;
          details.append(link);
        }
        details.append(tree);
        container.append(details);
        details.addEventListener('toggle', () => {
          const canvas = details.querySelector('.graph-scope:not([hidden]) .policy-canvas');
          if (details.open && canvas && !canvas.dataset.scale) fitGraphCanvas(canvas);
        });
        if (graphOpen && scopeExists && scale) requestAnimationFrame(() => {
          const canvas = details.querySelector('.graph-scope:not([hidden]) .policy-canvas');
          if (!canvas) return;
          fitGraphCanvas(canvas, Number(scale));
          [canvas.scrollLeft, canvas.scrollTop] = scroll;
        });
      }
      return;
    }
    container.replaceChildren();
    container.hidden = true;
  }};
}
