import { Euler, Quaternion } from 'three';
import './simulation-scene.css';

export function mountSimulationScene(root) {
  const document = root.ownerDocument;
  const byId = id => root.querySelector(`[data-id="${id}"]`);
  const shapes = { box: '长方体', sphere: '球', cylinder: '圆柱', capsule: '胶囊', ellipsoid: '椭球', mesh: 'CAD' };
  const colors = { 红色: '#e34444', 橙色: '#e88c30', 黄色: '#f0d44c', 绿色: '#36ad65', 蓝色: '#225bcf', 粉色: '#e64d8c', 紫色: '#9861cc', 黑色: '#080808', 白色: '#eeeeee', 灰色: '#999999', 棕色: '#735039' };
  let snapshot;
  let busy = false;

  async function request(path, body) {
    const response = await fetch('/api/simulation-scene' + path, body === undefined ? {} : {
      method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(typeof result.detail === 'string' ? result.detail : JSON.stringify(result.detail));
    return result;
  }

  function status(message, error = false) {
    byId('scene-status').textContent = message;
    byId('scene-status').className = error ? 'error' : '';
  }

  function controls() {
    byId('load-scene').disabled = busy;
    byId('scene-controls').disabled = busy || snapshot?.editable !== true;
  }

  async function perform(operation) {
    if (busy) return;
    busy = true; controls();
    try { await operation(); } catch (error) { status(error.message, true); }
    finally { busy = false; controls(); }
  }

  function field(parent, label, value, update, disabled = false) {
    const wrapper = document.createElement('label'); wrapper.textContent = label;
    const input = document.createElement('input');
    input.type = 'number'; input.step = 'any'; input.value = Number(value.toFixed(3)); input.disabled = disabled;
    input.addEventListener('change', () => {
      if (input.value.trim() === '' || !Number.isFinite(input.valueAsNumber)) {
        input.setCustomValidity('请输入有限数值'); input.reportValidity(); return;
      }
      input.setCustomValidity(''); update(input.valueAsNumber); status('有未应用的修改');
    });
    wrapper.append(input); parent.append(wrapper);
  }

  function checkbox(parent, label, checked, update, disabled = false) {
    const wrapper = document.createElement('label'); wrapper.className = 'check-field';
    const input = document.createElement('input'); input.type = 'checkbox'; input.checked = checked; input.disabled = disabled;
    input.addEventListener('change', () => { update(input.checked); render(); status('有未应用的修改'); });
    wrapper.append(input, label); parent.append(wrapper);
  }

  function colorFields(parent, item) {
    const label = document.createElement('label'); label.textContent = '物体颜色';
    const select = document.createElement('select');
    const custom = document.createElement('option'); custom.textContent = '自定义'; custom.value = ''; select.append(custom);
    const picker = document.createElement('input'); picker.type = 'color';
    picker.value = '#' + item.color_rgba.slice(0, 3).map(v => Math.round(v * 255).toString(16).padStart(2, '0')).join('');
    for (const [name, hex] of Object.entries(colors)) {
      const option = document.createElement('option'); option.value = hex; option.textContent = name; select.append(option);
    }
    const setColor = hex => { item.color_rgba = [1, 3, 5].map(i => parseInt(hex.slice(i, i + 2), 16) / 255).concat(1); status('有未应用的修改'); };
    select.value = Object.values(colors).includes(picker.value) ? picker.value : '';
    select.addEventListener('change', () => { if (select.value) { picker.value = select.value; setColor(select.value); } });
    picker.addEventListener('input', () => { select.value = ''; setColor(picker.value); });
    label.append(select, picker); parent.append(label);
  }

  function render() {
    const container = byId('scene-objects'); container.replaceChildren();
    if (snapshot?.editable !== true) return;
    const objects = snapshot.configuration.objects;
    byId('revision').textContent = `配置 ${snapshot.config_revision} · 已应用 ${snapshot.applied_revision} · ${objects.length} 个物体`;
    for (const item of objects) {
      const card = document.createElement('section'); card.className = 'object-card'; card.dataset.objectId = item.id;
      const heading = document.createElement('h2'); heading.textContent = `${item.label} · ${shapes[item.shape]}`;
      const remove = document.createElement('button'); remove.textContent = '删除';
      remove.addEventListener('click', () => { objects.splice(objects.indexOf(item), 1); render(); status('有未应用的修改'); });
      heading.append(remove); card.append(heading);
      const label = document.createElement('label'); label.textContent = '名称 ';
      const nameInput = document.createElement('input'); nameInput.value = item.label; nameInput.maxLength = 100;
      nameInput.addEventListener('change', () => { item.label = nameInput.value; status('有未应用的修改'); });
      label.append(nameInput); card.append(label);
      const grid = document.createElement('div'); grid.className = 'field-grid';
      ['X', 'Y', 'Z'].forEach((axis, i) => field(grid, `${axis} 位置`, item.pose_wxyz[i] * 1000, value => { item.pose_wxyz[i] = value / 1000; }));
      const [w, x, y, z] = item.pose_wxyz.slice(3);
      const angles = new Euler().setFromQuaternion(new Quaternion(x, y, z, w), 'ZYX');
      const values = [angles.x, angles.y, angles.z];
      ['X', 'Y', 'Z'].forEach((axis, i) => field(grid, `${axis} 旋转`, values[i] * 180 / Math.PI, value => {
        values[i] = value * Math.PI / 180;
        const q = new Quaternion().setFromEuler(new Euler(...values, 'ZYX'));
        item.pose_wxyz.splice(3, 4, q.w, q.x, q.y, q.z);
      }));
      const round = ['sphere', 'cylinder', 'capsule'].includes(item.shape);
      ['X 尺寸', 'Y 尺寸', 'Z 尺寸'].forEach((axis, i) => field(grid, round && i === 0 ? '直径' : axis, item.size_m[i] * 1000, value => {
        item.size_m[i] = value / 1000;
        if (round && i === 0) { item.size_m[1] = value / 1000; if (item.shape === 'sphere') item.size_m[2] = value / 1000; render(); }
      }, item.shape === 'mesh' || (round && i === 1) || (item.shape === 'sphere' && i === 2)));
      field(grid, '质量 kg', item.mass_kg, value => { item.mass_kg = value; }, item.fixed);
      colorFields(grid, item);
      card.append(grid);
      checkbox(card, '固定不动', item.fixed, value => { item.fixed = value; if (value) item.gravity_enabled = false; });
      checkbox(card, '启用重力', item.gravity_enabled, value => { item.gravity_enabled = value; }, item.fixed);
      const duplicate = document.createElement('button'); duplicate.textContent = '按上方数量复制';
      duplicate.addEventListener('click', () => perform(async () => { addCopies(item); })); card.append(duplicate);
      container.append(card);
    }
  }

  function uniqueName(prefix) {
    const existing = new Set(snapshot.configuration.objects.map(item => item.id));
    let index = 1; while (existing.has(`${prefix}_${index}`)) index += 1;
    return `${prefix}_${index}`;
  }

  function addCopies(template, count = byId('object-count').valueAsNumber) {
    if (!Number.isInteger(count) || count < 1 || snapshot.configuration.objects.length + count > 128) throw new Error('数量应为正整数，场景最多 128 个物体');
    for (let i = 0; i < count; i += 1) {
      const item = structuredClone(template); item.id = uniqueName(item.shape);
      item.pose_wxyz[1] += (i + (template.id ? 1 : 0)) * (Math.max(...item.size_m) + 0.03);
      snapshot.configuration.objects.push(item);
    }
    render(); status('物体已添加，可逐个调整颜色、位姿和重力，尚未应用');
  }

  async function load() {
    snapshot = undefined; controls();
    const result = await request(''); snapshot = result;
    if (!result.editable) {
      byId('connection-mode').textContent = result.runtime_mode === 'real' ? '当前连接：真机' : '当前设备不支持仿真场景';
      byId('revision').textContent = '场景编辑已禁用'; render(); status(result.message); return;
    }
    byId('connection-mode').textContent = '当前连接：仿真 · 自动识别所配置 Base 地址的设备';
    const catalog = await request('/cad-assets'); byId('cad-asset').replaceChildren();
    for (const asset of catalog.assets) {
      const option = document.createElement('option'); option.value = asset.id; option.textContent = asset.label; byId('cad-asset').append(option);
    }
    render(); status(result.apply_error || '场景已读取', Boolean(result.apply_error));
  }

  byId('load-scene').addEventListener('click', () => perform(load));
  byId('add-shape').addEventListener('click', () => perform(async () => {
    const shape = byId('shape').value;
    addCopies({ shape, label: shapes[shape], size_m: [.1, .1, shape === 'capsule' ? .2 : .1],
      pose_wxyz: [.5, 0, .75, 1, 0, 0, 0], color_rgba: [.6, .6, .6, 1], gravity_enabled: true, fixed: false, mass_kg: .2 });
  }));
  byId('add-table').addEventListener('click', () => perform(async () => {
    addCopies({ shape: 'box', label: '桌面', size_m: [.7, 1, .08], pose_wxyz: [.55, 0, .56, 1, 0, 0, 0],
      color_rgba: [.45, .3, .18, 1], gravity_enabled: false, fixed: true, mass_kg: 20 }, 1);
  }));
  byId('add-cad').addEventListener('click', () => perform(async () => {
    const item = await request('/cad-assets', { asset_id: byId('cad-asset').value }); addCopies(item);
  }));
  async function apply(resetRobot) {
    if (resetRobot) {
      snapshot = await request('/reset', { expected_revision: snapshot.config_revision });
    } else {
    if ([...root.querySelectorAll('input')].some(input => !input.checkValidity())) throw new Error('请修正无效输入');
    snapshot = await request('', { configuration: snapshot.configuration, expected_revision: snapshot.config_revision });
    }
    snapshot.editable = true; render(); status('已提交，等待物理线程应用');
    for (let attempt = 0; attempt < 60; attempt += 1) {
      await new Promise(resolve => setTimeout(resolve, 500));
      const current = await request('');
      if (!current.editable || current.config_revision !== snapshot.config_revision) throw new Error('设备或场景已变化，请重新读取');
      if (current.apply_error) throw new Error(current.apply_error);
      if (current.applied_revision === current.config_revision && current.ready === true) {
        snapshot = current; render(); status(resetRobot ? '仿真已复位，双臂与物体布局已恢复' : '场景已应用到仿真'); return;
      }
    }
    throw new Error('配置已提交，仿真尚未确认；请重新读取状态');
  }
  byId('apply-scene').addEventListener('click', () => perform(() => apply(false)));
  byId('reset-simulation').addEventListener('click', () => perform(() => apply(true)));
  let timer;
  return {setActive(active) {
    clearInterval(timer);
    if (!active) return;
    if (!snapshot?.editable && !busy) void perform(load);
    timer = setInterval(() => {
      if (!snapshot?.editable && !busy) void perform(load);
    }, 2500);
  }};
}
