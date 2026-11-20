import './workbench.css';
import {request} from './api.js';

const modules = {
  scene: async root => (await import('./simulation-scene.js')).mountSimulationScene(root),
  spheres: async root => (await import('./collision-sphere-page.js')).mountCollisionSpheres(root),
  workspaces: async root => (await import('./workspace-page.js')).mountWorkspaces(root),
  replay: async (root, params) => (await import('./replay.js')).mountReplayPage(root, params),
  base: async root => {
    const manifest = await request('/api/manifest');
    const frame = root.querySelector('iframe');
    frame.src = manifest.base_url;
    return {setActive() {}};
  },
};

export function mountWorkbench({onTaskVisibility}) {
  const mounted = new Map();
  const pending = new Map();
  let current = 'task';
  let navigation = 0;
  const media = document.getElementById('media-preview');
  document.getElementById('close-media-preview').addEventListener('click', () => media.close());
  document.addEventListener('click', event => {
    const link = event.target.closest('a[data-preview-image]');
    if (!link) return;
    event.preventDefault();
    const img = media.querySelector('img');
    img.src = link.href;
    img.alt = link.querySelector('img')?.alt || '实验归档图片';
    media.showModal();
  });

  async function navigate() {
    const operation = ++navigation;
    const [name, query = ''] = location.hash.slice(1).split('?');
    current = Object.hasOwn(modules, name) ? name : 'task';
    const params = new URLSearchParams(query);
    document.getElementById('workspace').hidden = current !== 'task';
    onTaskVisibility(current === 'task');
    for (const [id, module] of mounted) module.setActive(id === current);
    for (const id of Object.keys(modules)) document.getElementById(`panel-${id}`).hidden = id !== current;
    for (const link of document.querySelectorAll('[data-panel]')) {
      if (link.dataset.panel === current) link.setAttribute('aria-current', 'page');
      else link.removeAttribute('aria-current');
    }
    if (current === 'task') return;
    const id = current;
    const root = document.getElementById(`panel-${id}`);
    try {
      if (!mounted.has(id)) {
        if (!pending.has(id)) pending.set(id, modules[id](root, params).then(module => {
          mounted.set(id, module);
          module.setActive(current === id);
          return module;
        }).finally(() => pending.delete(id)));
        await pending.get(id);
      }
      if (operation !== navigation) return;
      mounted.get(id).setActive(true);
      await mounted.get(id).navigate?.(params);
      root.querySelector('.workbench-error')?.remove();
    } catch (error) {
      let message = root.querySelector('.workbench-error');
      if (!message) { message = document.createElement('p'); message.className = 'workbench-error'; message.setAttribute('role', 'alert'); root.prepend(message); }
      message.textContent = `加载失败：${error.message}。切换后可重试。`;
    }
  }
  window.addEventListener('hashchange', navigate);
  window.addEventListener('pagehide', () => {
    onTaskVisibility(false);
    for (const module of mounted.values()) module.setActive(false);
  });
  window.addEventListener('pageshow', event => { if (event.persisted) void navigate(); });
  void navigate();
}
