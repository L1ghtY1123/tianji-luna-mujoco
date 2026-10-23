// Bounded recovery for a page opened while the backend is restarting.
export async function loadCadAssets(request, wait = ms => new Promise(resolve => setTimeout(resolve, ms))) {
  for (let attempt = 0; attempt < 3; attempt++) {
    try {
      const payload = await request('/api/cad-assets');
      if (!Array.isArray(payload?.assets) || !payload.assets.length) throw new Error('CAD 资产目录格式无效');
      return payload.assets;
    } catch (error) {
      if (attempt === 2) throw error;
      await wait(1000 * (attempt + 1));
    }
  }
}
