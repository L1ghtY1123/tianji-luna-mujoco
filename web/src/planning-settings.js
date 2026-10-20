const GROUPS = {
  shared: ['共享 IK 参数', '机械臂评估与 plan_grasp_batch 共用 IK 参数。IK 优化 seeds 与 LM 初始解 seeds 分别控制优化起点数和初始解生成预算；random seed 是复现随机序列的编号，不是搜索数量。'],
  planner: ['plan_grasp_batch', 'TrajOpt seeds 是轨迹优化的初始轨迹数量。它与 IK 优化 seeds、LM 初始解 seeds 属于三个不同阶段的搜索预算。'],
  grasp: ['抓取几何与候选', '设置抓取候选和接近、抓取、提升的几何条件。'],
  place: ['放置位姿', '两指前端 TCP 的世界 XY 固定为分区中心，绕世界 Z 轴生成 18 个 yaw。attachment 球底规划间隙为碰撞阈值加 2 mm；实际不得低于碰撞阈值，默认允许 10–14 mm。'],
  filter: ['可见点云过滤（默认关闭）', '只处理 RGBD 实测点；统计或半径过滤不能保证去除密集浮岛。DBSCAN 仅诊断点簇，不按点簇大小或桌面连通性删除障碍。'],
  observation: ['空手观察位姿 · 只读', 'A、B 臂各 7 个已标定参考关节角，用于计算观察 TCP 目标。校准值在此只读；运行仍须通过碰撞规划与 RGBD 覆盖检查。'],
  attachment: ['持物可见表面碰撞球', '修改应用于下一次抓取，当前持物绑定保持原参数。仅在提升后 verify_grasp 确认持物后建立。体素大小、附加余量和数量预算只作用于实测可见表面，不补齐遮挡面。超预算时调整表面采样并报告实际分辨率，保留全部实测点的覆盖。'],
  verification: ['持物位置判断', '完整闭合命令后比较实测位置与空夹基准；不使用力、电流或驱动状态。'],
  execution: ['执行时间', 'time_scale 统一作用于抓取与放置轨迹；数值越大，执行越慢。停留时间不乘倍率。'],
};

const CATEGORIES = {
  common: ['常用', '执行倍率与阶段停留。修改后需应用，下一次任务使用。'],
  experimental: ['实验调参', '解锁后可修改求解预算、几何、坐标轴与过滤；当前生效值不会被默认值覆盖。'],
  system: ['系统只读', '自碰撞状态与机器人观察标定保持当前生效值，网页不提供修改入口。'],
};

export function planningFieldCategory(name, field) {
  if (name === 'self_collision_check' || name === 'place_release_height_m' || field.group === 'observation' || name.startsWith('observation.')) return 'system';
  return field.group === 'execution' ? 'common' : 'experimental';
}

function choiceLabel(value, field) {
  if (field.choices?.[String(value)]) return field.choices[String(value)];
  if (typeof value === 'boolean') return value ? '启用' : '关闭';
  return displayPlanningValue(value, field);
}

export function planningFieldMetadata(name, field) {
  const unit = field.unit ? ` ${field.unit}` : '';
  const bounds = [];
  for (const [key, label] of [['minimum', '≥'], ['exclusiveMinimum', '>'], ['maximum', '≤'], ['exclusiveMaximum', '<']]) {
    if (field[key] !== undefined) bounds.push(`${label} ${displayPlanningValue(field[key], field)}`);
  }
  const choices = field.type === 'boolean' ? [true, false] : field.enum;
  const descriptions = {
    ik_num_seeds: '每目标 IK 优化起点数；增大搜索预算通常增加计算量，两处 IK 使用相同设置。',
    ik_seed_solver_num_seeds: 'LM 初始解搜索预算；项目要求不小于 IK seeds。',
    trajopt_num_seeds: '每目标轨迹优化起点数；项目要求不大于 IK seeds。',
    self_collision_check: '当前机器人自身碰撞检查状态；此处只读。',
  };
  // Verified against the installed, pinned cuRobo SDK. These are interface-specific.
  const officialDefaults = {
    ik_num_seeds: 'cuRobo batch 官方默认 16；独立 IK 默认 32，本项目显式对齐为 16',
    ik_seed_solver_num_seeds: 'cuRobo LM 官方默认 32',
    trajopt_num_seeds: 'cuRobo batch 官方默认 2；单任务接口默认 4',
    ik_position_threshold_m: 'cuRobo IK 官方默认 5 mm',
    ik_rotation_threshold_rad: 'cuRobo IK 官方默认 0.05 rad（约 2.865°）',
  };
  return {
    defaultValue: field.default === undefined ? '未由 schema 提供' : `${choiceLabel(field.default, field)}${unit}`,
    range: choices ? choices.map(value => choiceLabel(value, field)).join(' / ') : bounds.length ? `${bounds.join('，')}${unit}` : '由后端校验',
    source: planningFieldCategory(name, field) === 'system' && name !== 'self_collision_check'
      ? '项目标定 resources/observation_poses.json；范围来自机器人 URDF'
      : `项目 schema · src/jev_mujoco/planning_settings.py${officialDefaults[name] ? `；${officialDefaults[name]}（锁定 SDK）` : '；项目采用值，不代表所有官方接口默认'}`,
    impact: descriptions[name] ?? field.description ?? '应用后供下一次任务使用；Reset 保留已应用值。',
  };
}

export function displayPlanningValue(value, field) {
  if (typeof value !== 'number') return String(value);
  if (!Number.isFinite(value)) return '';
  return String(Number((value * (field.display_scale ?? 1)).toFixed(8)));
}

export function parsePlanningValue(text, field) {
  const label = field.title ?? '参数';
  if (field.type === 'boolean') {
    if (text === 'true') return true;
    if (text === 'false') return false;
    throw new Error(`${label}：请选择启用或关闭。`);
  }
  if (field.enum) {
    const selected = field.enum.find(value => String(value) === text);
    if (selected === undefined) throw new Error(`${label}：请选择允许的选项。`);
    return selected;
  }
  const value = text.trim() === '' ? NaN : Number(text) / (field.display_scale ?? 1);
  if (!Number.isFinite(value)) throw new Error(`${label}：必须为有限数值。`);
  if (field.type === 'integer' && !Number.isInteger(value)) throw new Error(`${label}：必须为整数。`);
  const unit = field.unit ? ` ${field.unit}` : '';
  if (field.minimum !== undefined && value < field.minimum) throw new Error(`${label}：不得小于 ${displayPlanningValue(field.minimum, field)}${unit}。`);
  if (field.maximum !== undefined && value > field.maximum) throw new Error(`${label}：不得大于 ${displayPlanningValue(field.maximum, field)}${unit}。`);
  if (field.exclusiveMinimum !== undefined && value <= field.exclusiveMinimum) throw new Error(`${label}：必须大于 ${displayPlanningValue(field.exclusiveMinimum, field)}${unit}。`);
  if (field.exclusiveMaximum !== undefined && value >= field.exclusiveMaximum) throw new Error(`${label}：必须小于 ${displayPlanningValue(field.exclusiveMaximum, field)}${unit}。`);
  return value;
}

export function createPlanningSettingsEditor(container, { document: ownerDocument = globalThis.document, onChange = () => {} } = {}) {
  let schema = null;
  let latest = null;
  let revision = null;
  let dirty = false;
  let conflict = false;
  let advancedLocked = true;
  let category = 'common';
  let applied = {};
  const inputs = new Map();
  let schemaEntries = [];

  function schemaFields() {
    const result = [];
    function visit(name, field) {
      const nested = field.$ref ? schema.$defs?.[field.$ref.split('/').at(-1)] : field;
      if (nested?.properties) {
        for (const [childName, childField] of Object.entries(nested.properties)) {
          visit(name ? `${name}.${childName}` : childName, childField);
        }
      } else if (nested?.prefixItems) {
        nested.prefixItems.forEach((item, index) => visit(`${name}.${index}`, item));
      } else {
        result.push([name, nested]);
      }
    }
    visit('', schema);
    return result;
  }

  function fieldValue(settings, name) {
    return name.split('.').reduce((value, part) => value?.[part], settings);
  }

  function setFieldValue(settings, name, value) {
    const path = name.split('.');
    let parent = settings;
    for (let index = 0; index < path.length - 1; index += 1) {
      parent[path[index]] ??= /^\d+$/.test(path[index + 1]) ? [] : {};
      parent = parent[path[index]];
    }
    parent[path.at(-1)] = value;
  }

  function element(tag, className, text) {
    const node = ownerDocument.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function render() {
    if (!schema || !latest) return;
    inputs.clear();
    schemaEntries = schemaFields();
    const categories = new Map(Object.entries(CATEGORIES).map(([key, [title, description]]) => {
      const section = element('section', 'planning-category');
      section.id = `planning-category-${key}`;
      section.hidden = category !== key;
      section.append(element('h2', '', title), element('p', 'planning-category-note', description));
      return [key, section];
    }));
    for (const categoryName of categories.keys()) {
      for (const [group, [title, description]] of Object.entries(GROUPS)) {
        const section = element('section', 'planning-group');
        section.append(element('h3', '', title), element('p', 'planning-group-note', description));
        const fields = element('div', 'planning-grid');
        for (const [name, field] of schemaEntries) {
          if ((field.group ?? 'planner') !== group || planningFieldCategory(name, field) !== categoryName) continue;
          const readOnly = categoryName === 'system';
          const label = element('label', 'input-field planning-field');
          label.append(element('span', '', `${field.title ?? name}${field.unit ? ` / ${field.unit}` : ''}`));
          const choices = field.type === 'boolean' ? [true, false] : field.enum;
          const input = element(readOnly ? 'output' : choices ? 'select' : 'input');
          input.name = name;
          input.disabled = readOnly || (advancedLocked && group !== 'execution');
          if (readOnly) {
            input.textContent = choiceLabel(fieldValue(latest.settings, name), field);
          } else if (choices) {
            for (const value of choices) {
              const text = field.choices?.[String(value)] ?? (value === true ? '启用' : value === false ? '关闭' : String(value));
              const option = element('option', '', text);
              option.value = String(value);
              input.append(option);
            }
          } else {
            input.type = 'number';
            input.inputMode = field.type === 'integer' ? 'numeric' : 'decimal';
            input.step = String(field.step ?? (field.type === 'integer' ? 1 : 'any'));
            if (field.minimum !== undefined) input.min = displayPlanningValue(field.minimum, field);
            if (field.maximum !== undefined) input.max = displayPlanningValue(field.maximum, field);
          }
          input.value = readOnly ? choiceLabel(fieldValue(latest.settings, name), field) : displayPlanningValue(fieldValue(latest.settings, name), field);
          if (!readOnly) input.addEventListener('input', () => { dirty = true; onChange(); });
          inputs.set(name, input);
          label.append(input);
          const metadata = planningFieldMetadata(name, field);
          label.append(element('p', 'planning-field-note', `修改影响：${metadata.impact}`));
          const reference = element('details', 'planning-reference');
          reference.append(element('summary', '', `项目默认：${metadata.defaultValue}`),
            element('p', '', `范围：${metadata.range}`), element('p', '', `来源：${metadata.source}`),
            element('code', '', name));
          label.append(reference);
          fields.append(label);
        }
        if (!fields.children.length) continue;
        section.append(fields);
        categories.get(categoryName).append(section);
      }
    }
    container.replaceChildren(...categories.values());
    revision = latest.revision;
    applied = { ...latest.settings };
    dirty = false;
    conflict = false;
  }

  return {
    get ready() { return Boolean(schema && revision); },
    get dirty() { return dirty; },
    get conflict() { return conflict; },
    get revision() { return revision; },
    get advancedLocked() { return advancedLocked; },
    setCategory(value) {
      if (!Object.hasOwn(CATEGORIES, value)) return;
      category = value;
      for (const section of container.children) section.hidden = section.id !== `planning-category-${category}`;
    },
    setAdvancedLocked(locked) {
      advancedLocked = locked;
      for (const [name, field] of schemaEntries) {
        inputs.get(name).disabled = planningFieldCategory(name, field) === 'system' || (advancedLocked && field.group !== 'execution');
      }
    },
    load(payload) {
      if (!payload?.schema?.properties || !payload.settings || !payload.revision) throw new Error('算法参数结构不完整，请重新读取。');
      schema = payload.schema;
      // A state snapshot may arrive while the initial schema request is pending.
      latest ??= payload;
      render();
    },
    sync(payload) {
      if (!payload?.settings || !payload.revision) return;
      latest = payload;
      if (dirty) { conflict ||= payload.revision !== revision; return; }
      if (payload.revision !== revision) render();
    },
    accept(payload) {
      if (!payload?.settings || !payload.revision) throw new Error('算法参数响应不完整，请重新读取。');
      if (payload.schema?.properties) schema = payload.schema;
      latest = payload;
      render();
    },
    markConflict() { conflict = true; },
    read() {
      if (!schema || !revision) throw new Error('算法参数尚未加载，请重新读取。');
      if (conflict) throw new Error('其他页面已更新算法参数，请重新读取后检查。');
      const values = {};
      for (const [name, field] of schemaEntries) {
        const text = inputs.get(name).value;
        // Preserve untouched SI values before parsing: degree rounding can cross a joint limit.
        const appliedValue = fieldValue(applied, name);
        const selected = planningFieldCategory(name, field) === 'system' || text === displayPlanningValue(appliedValue, field)
          ? appliedValue : parsePlanningValue(text, field);
        setFieldValue(values, name, selected);
      }
      return values;
    },
  };
}
