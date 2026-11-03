import { isTaskInventory, renderTaskInventory } from './task-inventory.js';
import { renderPolicyGraph } from './replay-policy.js';

export function labelsForFrame(labels, frameId, generation, cameraId = '3') {
  return labels?.observation_id === frameId && labels.generation === generation && Array.isArray(labels.items)
    ? labels.items.filter(item => (item.camera_id ?? '3') === cameraId && typeof item.id === 'string'
      && Number.isFinite(item.x) && Number.isFinite(item.y)) : [];
}

export function createObjectList(container, {document: ownerDocument = globalThis.document} = {}) {
  let graphDetails = null;
  return {render(progress, bindings = null, policyGraph = null, executionEvents = []) {
    const graphOpen = graphDetails?.open ?? false;
    graphDetails = null;
    if (isTaskInventory(progress)) {
      container.replaceChildren();
      container.hidden = false;
      renderTaskInventory(container, progress, bindings);
      if (policyGraph) {
        const details = ownerDocument.createElement('details');
        details.open = graphOpen;
        graphDetails = details;
        const title = ownerDocument.createElement('summary');
        title.textContent = 'Responses 完整任务图';
        const tree = ownerDocument.createElement('div');
        tree.className = 'policy-tree';
        const versionStart = executionEvents.findLastIndex(event => ['skill_program', 'policy_revised'].includes(event.kind));
        renderPolicyGraph(tree, policyGraph, executionEvents.slice(versionStart + 1));
        details.append(title, tree);
        container.append(details);
      }
      return;
    }
    container.replaceChildren();
    container.hidden = true;
  }};
}
