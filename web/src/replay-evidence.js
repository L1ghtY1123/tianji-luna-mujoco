import { modelCallInfo, modelResult } from './replay-story.js';

const STAGE_NAMES = { raw_graspgen: '① GraspGen 全部生成位姿', before_ik: '② 方向与执行臂预筛选', after_ik: '③ Batched IK 筛选', after_plan_grasp_batch: '④ plan_grasp_batch 筛选' };
const STATUS_NAMES = { passed: '通过', failed: '未通过', unknown: '未记录判定' };

export function safeArchivePath(path) {
  return typeof path === 'string' && path.length > 0 && !/[\\:\0]/.test(path)
    && path.split('/').every(part => part && part !== '.' && part !== '..');
}

export function extractReplayEvidence(value) {
  const result = { images: [], segmentation: [], candidates: [], models: [] };
  const seen = new Set();
  function visit(item) {
    if (!item || typeof item !== 'object' || seen.has(item)) return;
    seen.add(item);
    if (Array.isArray(item.request?.input)) {
      if (item.endpoint && item.response) result.models.push(item);
      let label = item.endpoint ?? '模型输入';
      let frame = 'other';
      for (const message of item.request.input) {
        for (const part of message.content ?? []) {
          if (part.type === 'input_text') {
            try {
              const text = JSON.parse(part.text);
              if (text.image_role) {
                const camera = text.camera_id ?? text.source_camera_id ?? text.camera;
                frame = text.frame_role === 'CURRENT' || text.image_role.startsWith('current_') ? 'current'
                  : text.frame_role === 'BEFORE' || text.image_role.startsWith('initial_') ? 'reference' : 'other';
                const roles = { initial_reference_annotations: '初始参考 · 身份标注', current_work_area_zoom: '当前 · 工作区放大',
                  drop_bin_region_zoom: `${frame === 'current' ? '当前' : '投放前'} · 盒内放大` };
                label = [camera && `相机 ${camera}`, roles[text.image_role] ?? text.image_role].filter(Boolean).join(' · ');
              }
            } catch {
              // Camera labels in older archives are plain text, not JSON.
              if (part.text.length < 240 && /camera\s+\d/i.test(part.text)) {
                const camera = /camera\s+(\d)/i.exec(part.text)?.[1];
                frame = /CURRENT/i.test(part.text) ? 'current' : /INITIAL|BEFORE/i.test(part.text) ? 'reference' : 'other';
                const when = frame === 'current' ? '当前' : /INITIAL/i.test(part.text) ? '初始参考' : frame === 'reference' ? '动作前' : '观测';
                label = `相机 ${camera} · ${when} · ${/cyan region/i.test(part.text) ? '区域标注' : '原始 RGB'}`;
              }
            }
          }
          if (part.type === 'input_image' && safeArchivePath(part.image_url?.path)) {
            result.images.push({ ...part.image_url, label, frame, endpoint: item.endpoint, observationId: item.observation_id });
          }
        }
      }
    }
    if (Array.isArray(item.sam3_queries)) result.segmentation.push(...item.sam3_queries);
    if (item.candidate_replay && typeof item.candidate_replay === 'object') {
      if (item.candidate_replay.stage === 'before_ik' && item.candidate_replay.all_generated_candidates?.length) {
        result.candidates.push({ ...item.candidate_replay, stage: 'raw_graspgen' });
      }
      result.candidates.push({ ...item.candidate_replay,
        candidate_indices: item.candidate_replay.candidate_indices ?? item.candidate_indices,
        candidate_filters: item.candidate_replay.candidate_filters === undefined
          ? item.candidate_filters : item.candidate_replay.candidate_filters });
    }
    for (const child of Object.values(item)) visit(child);
  }
  visit(value);
  return result;
}

function ikEligibility(row, replay) {
  const result = { ...row, rawStatus: row.status, eligible: null, filterEvidence: null };
  const filters = replay.candidate_filters;
  const reason = detail => [row.reason, detail].filter(Boolean).join(' · ');
  if (filters === undefined) return { ...result,
    reason: reason('后置筛选未归档；仅原始 IK 结果，最终可用性未记录') };
  const validIndex = value => Number.isInteger(value) && value >= 0;
  const invalidList = !Array.isArray(filters) || filters.some(filter => !validIndex(filter?.candidate_index));
  const invalidMapping = !Array.isArray(replay.candidate_indices) || !validIndex(row.index)
    || replay.candidate_indices.filter(index => index === row.index).length !== 1;
  const matches = Array.isArray(filters) ? filters.filter(filter => filter?.candidate_index === row.index) : [];
  result.filterEvidence = invalidList || invalidMapping ? filters : matches.length === 1 ? matches[0] : matches;
  if (invalidList || invalidMapping) return { ...result, status: 'unknown',
    reason: reason('筛选记录或原始候选索引损坏，最终可用性未知') };
  // Only successful raw IK candidates receive post-IK filters in the archive.
  if (!matches.length && row.status === 'failed') return { ...result, eligible: false,
    reason: reason('原始 IK 未通过，未进行后置筛选') };
  if (matches.length !== 1 || typeof matches[0].accepted !== 'boolean') return { ...result, status: 'unknown',
    reason: reason(matches.length > 1 ? '筛选记录重复，最终可用性未知' : '筛选记录缺失或损坏，最终可用性未知') };
  const filter = matches[0];
  const eligible = !filter.accepted || row.status === 'failed' ? false : row.status === 'passed' ? true : null;
  return { ...result, eligible, status: eligible === null ? 'unknown' : eligible ? 'passed' : 'failed',
    reason: reason(typeof filter.reason === 'string' ? filter.reason : filter.accepted ? '后置筛选通过' : '后置筛选未通过') };
}

export function candidateRows(replay) {
  if (replay.stage === 'raw_graspgen') {
    return (replay.all_generated_candidates ?? []).map(candidate => ({
      index: candidate.index, score: candidate.score, matrix: candidate.T_world_tcp, status: 'unknown',
      reason: replay.filter_all_generated || candidate.index < replay.requested_candidate_count
        ? '进入本轮预筛选' : '超出本轮采样池预算，未送入预筛选',
      sourceBatch: candidate.source_batch, sourceIndex: candidate.source_index,
    }));
  }
  if (replay.stage === 'before_ik') {
    const evidence = new Map((replay.direction_filter?.candidate_evidence ?? []).map(row => [row.candidate_index, row]));
    return (replay.candidates_before_ik ?? []).map(candidate => {
      const row = evidence.get(candidate.index);
      const evaluated = typeof row?.tilt_passed === 'boolean' && typeof row?.away_from_robot_passed === 'boolean';
      const dot = value => Number.isFinite(value) ? value.toFixed(4) : '未记录';
      const budgetLimited = replay.pre_budget_eligible_indices?.includes(candidate.index)
        && !replay.pre_ik_eligible_indices?.includes(candidate.index);
      return { index: candidate.index, score: candidate.score, matrix: candidate.T_world_tcp,
        beforeInsertion: candidate.T_world_tcp_before_insertion,
        status: budgetLimited ? 'unknown' : Array.isArray(replay.pre_ik_eligible_indices)
          ? replay.pre_ik_eligible_indices.includes(candidate.index) ? 'passed' : 'failed'
          : evaluated ? row.tilt_passed && row.away_from_robot_passed ? 'passed' : 'failed' : 'unknown',
        reason: (row ? `向下 ${dot(row.downward_dot)} · 离机器人 ${dot(row.horizontal_away_from_robot_dot)}` : '')
          + (budgetLimited ? ' · 预筛选通过，超出本轮 IK 数量上限，未提交' : ''),
      };
    });
  }
  const items = new Map((replay.response?.items ?? []).map(item => [item.candidate_index, item]));
  return (replay.request?.goal_matrices_world ?? []).map((matrix, localIndex) => {
    const item = items.get(localIndex);
    const row = { index: replay.candidate_indices?.[localIndex] ?? localIndex, localIndex, matrix,
      status: item?.success === true ? 'passed' : item?.success === false ? 'failed' : 'unknown',
      reason: item?.error ?? item?.status ?? '', result: item };
    return replay.stage === 'after_ik' ? ikEligibility(row, replay) : row;
  });
}

export function candidateCloudSources(replay) {
  const sources = [];
  const available = reference => Array.isArray(reference) || reference?.type === 'ndarray';
  const composition = {
    height_completed: ['测量 + 先验补全', '真实提交数组包含测量点和按高度先验构造的点，不是全部实测。'],
    measured: ['实测点', '真实提交数组使用测量点；按归档的 T_world_cloud 还原世界坐标。'],
  }[replay.graspgen_input_source] ?? ['构成未归档', '使用归档的真实请求数组；输入构成未归档，无法断言全部为实测。'];
  if (replay.T_world_cloud) {
    for (const [id, title, key] of [
      ['graspgen-object', 'GraspGen 实际物体输入', 'object_points_camera_m'],
      ['graspgen-scene', 'GraspGen 实际场景输入', 'scene_points_camera_m'],
    ]) {
      const reference = replay.graspgen_request?.[key];
      if (available(reference)) sources.push({
        id, label: id === 'graspgen-object' ? `${title}（${composition[0]}）` : title,
        description: id === 'graspgen-object' ? composition[1]
          : '真实提交的场景请求数组，按归档的 T_world_cloud 还原世界坐标；物体输入的高度补全标记不应用于此场景数组。',
        reference, transform: replay.T_world_cloud,
      });
    }
  }
  if (available(replay.measured_points_world_m)) sources.push({
    id: 'measured', label: '原始测量目标点云',
    description: '原始目标测量点，未加入高度先验构造点；与实际 GraspGen 请求数组分开查看。',
    reference: replay.measured_points_world_m, transform: null,
  });
  if (available(replay.completed_geometry_points_world_m)) sources.push({
    id: 'completed', label: '补全几何全集（测量 + 先验补全）',
    description: '包含测量前缀和先验构造点的完整几何集合；不是全部实测，也不替代实际 GraspGen 请求数组。',
    reference: replay.completed_geometry_points_world_m, transform: null,
  });
  return sources;
}

export function createReplayEvidence(container, { request, fileUrl, document = container.ownerDocument, onFile } = {}) {
  let revision = 0;
  let cloudOperation = 0;
  const viewers = new Map();
  const cloudOperations = new Map();
  const element = (tag, text = '', className = '') => {
    const node = document.createElement(tag);
    node.textContent = text;
    node.className = className;
    return node;
  };
  const section = (title, open = false) => {
    const details = element('details', '', 'evidence-section');
    details.open = open;
    details.append(element('summary', title));
    container.append(details);
    return details;
  };
  function clear() {
    revision += 1;
    for (const viewer of viewers.values()) viewer.dispose();
    viewers.clear();
    cloudOperations.clear();
    container.replaceChildren();
  }
  async function array(runId, reference) {
    if (Array.isArray(reference)) return { data: reference, sampled: false, shape: [reference.length] };
    if (reference?.type !== 'ndarray' || !safeArchivePath(reference.path)) throw new Error('此历史记录没有可读取的数组证据');
    const query = new URLSearchParams({ path: reference.path, key: reference.key });
    return request(`/api/experiments/${encodeURIComponent(runId)}/arrays?${query}`);
  }
  async function renderMask(runId, proposal, parent, operation) {
    const message = element('p', '正在读取原始掩码…', 'muted');
    parent.append(message);
    try {
      const payload = await array(runId, proposal.mask);
      if (operation !== revision) return;
      const rows = payload.data;
      const height = rows.length, width = rows[0]?.length;
      if (!width || !height || payload.sampled) throw new Error('掩码尺寸无效');
      const canvas = element('canvas', '', 'evidence-mask');
      canvas.width = width; canvas.height = height;
      canvas.setAttribute('aria-label', 'SAM3 原始二值掩码，粉色为提议区域');
      const context = canvas.getContext('2d');
      const pixels = context.createImageData(width, height);
      for (let y = 0; y < height; y += 1) {
        for (let x = 0; x < width; x += 1) {
          const offset = (y * width + x) * 4;
          const active = Boolean(rows[y][x]);
          pixels.data.set(active ? [235, 113, 174, 255] : [12, 22, 32, 255], offset);
        }
      }
      context.putImageData(pixels, 0, 0);
      message.textContent = `${width} × ${height} 原始掩码 · 提议不代表已确认身份`;
      parent.append(canvas);
    } catch (error) { if (operation === revision) message.textContent = `掩码不可用：${error.message}`; }
  }
  async function renderCloud(runId, cloudReference, rows, parent, operation, transform = null, sourceLabel = '') {
    const cloudRevision = ++cloudOperation;
    cloudOperations.set(parent, cloudRevision);
    const message = element('p', '正在读取点云…', 'muted');
    const viewport = element('div', '', 'evidence-cloud');
    parent.append(message, viewport);
    try {
      const [payload, module] = await Promise.all([array(runId, cloudReference), import('./replay-cloud.js')]);
      if (operation !== revision || cloudRevision !== cloudOperations.get(parent)) return null;
      viewers.get(parent)?.dispose();
      const points = transform ? payload.data.map(point => transform.slice(0, 3).map(row =>
        row[0] * point[0] + row[1] * point[1] + row[2] * point[2] + row[3])) : payload.data;
      const activeViewer = module.createReplayCloud(viewport, { points, candidates: rows });
      viewers.set(parent, activeViewer);
      const shown = payload.data.length;
      message.textContent = `${sourceLabel ? sourceLabel + '。' : ''}世界坐标 / 米 · 显示 ${shown} / ${payload.shape[0]} 点${payload.sampled ? ' · 按原始索引均匀抽样' : ''} · 拖动旋转，右键平移，滚轮缩放`;
      const home = element('button', '适合点云', 'button');
      home.addEventListener('click', () => activeViewer.fit());
      parent.append(home);
      return activeViewer;
    } catch (error) { if (operation === revision) message.textContent = `3D 证据不可用：${error.message}`; }
  }
  async function show(runId, events, { values = [] } = {}) {
    clear();
    const operation = revision;
    const status = element('p', '正在读取节点证据…', 'muted');
    container.append(status);
    const paths = [...new Set(events.flatMap(event => [
      event.artifact, event.data?.artifact, event.data?.observation_artifact,
    ]).filter(path => safeArchivePath(path) && path.endsWith('.json')))];
    const failures = [];
    const loaded = await Promise.all(paths.map(async path => {
      try { return await request(fileUrl(runId, path)); }
      catch (error) { failures.push(`${path}: ${error.message}`); return null; }
    }));
    if (operation !== revision) return;
    const evidence = extractReplayEvidence([...values, ...loaded]);
    status.textContent = [evidence.images.length && `${evidence.images.length} 张实际模型输入`, evidence.segmentation.length && `${evidence.segmentation.length} 次 SAM3 查询`, evidence.candidates.length && `${evidence.candidates.length} 份候选结果`].filter(Boolean).join(' · ') || '此节点没有归档图片、分割或候选数据。';
    if (failures.length) container.append(element('p', `部分证据读取失败：${failures.join('；')}`, 'inline-warning'));
    for (const archive of evidence.models) {
      const info = modelCallInfo(archive);
      const result = modelResult(archive);
      const model = section(`${info.type} · ${info.purpose}`, true);
      if (info.failed) model.append(element('p', `调用失败：${archive.error ?? `HTTP ${archive.http_status}`}`, 'inline-error'));
      if (result.inventory) {
        const inventory = result.inventory;
        model.append(element('p', inventory.task?.reason ?? '模型已返回结构化场景与程序。'));
        const entities = element('details');
        entities.append(element('summary', `Responses 提议的实体 · ${inventory.entities?.length ?? 0} 项`));
        for (const entity of inventory.entities ?? []) entities.append(element('p', `${entity.ref} · ${entity.description ?? entity.kind}`));
        const program = element('details');
        program.append(element('summary', '查看编排程序（包含循环与条件）'), element('pre', JSON.stringify(inventory.program, null, 2), 'candidate-detail'));
        model.append(entities, program);
      }
      if (result.answers.length) {
        model.append(element('p', `${result.answers.length} 个问题 · 以下是模型原始返回，不等同于整段动作的最终验收。`, 'muted'));
        const answers = result.answers.length > 4 ? element('details') : element('div');
        if (result.answers.length > 4) answers.append(element('summary', `展开 ${result.answers.length} 项身份判断`));
        for (const answer of result.answers) {
          const row = element('article', '', 'model-answer');
          row.append(element('strong', answer.title));
          if (answer.type === 'predicate') row.append(element('p', `该命题成立的概率：${Number.isFinite(answer.probability) ? `${(answer.probability * 100).toFixed(1)}%` : '未归档'}`, 'answer-value'));
          else row.append(element('p', `选择：${answer.choiceLabel ?? '未归档'}${Number.isFinite(answer.confidence) ? ` · 返回 confidence ${answer.confidence}` : ''}`, 'answer-value'));
          const question = element('details');
          question.append(element('summary', '完整问题与原始回答'), element('p', answer.instructions),
            element('pre', JSON.stringify(archive.response.answers.find(item => item.name === answer.name), null, 2), 'candidate-detail'));
          row.append(question);
          answers.append(row);
        }
        model.append(answers);
      } else if (!result.inventory) model.append(element('p', '完整结构化回答未归档；请检查原始 JSON。', 'muted'));
    }
    const images = evidence.images.length ? section(`实际输入图片 · ${evidence.images.length} 张`, true) : element('div');
    const grid = element('div', '', 'evidence-images');
    const filter = element('select');
    filter.setAttribute('aria-label', '模型输入图片时刻');
    const frames = new Set(evidence.images.map(image => image.frame));
    for (const [value, title] of [['all', '全部实际输入（按发送顺序）'], ['current', '当前画面'], ['reference', '初始 / 动作前参考'], ['other', '其他观测输入']]) {
      if (value !== 'all' && !frames.has(value)) continue;
      const option = element('option', title); option.value = value; filter.append(option);
    }
    filter.value = frames.has('current') ? 'current' : 'all';
    const figures = [];
    for (const [index, image] of evidence.images.entries()) {
      const figure = element('figure');
      const link = element('a');
      link.href = fileUrl(runId, image.path); link.target = '_blank'; link.rel = 'noopener';
      const img = element('img'); img.src = link.href; img.alt = `${image.label} · 输入 ${index + 1}`; img.loading = 'lazy';
      link.append(img);
      const metadata = element('details');
      metadata.append(element('summary', '图片来源与校验值'), element('p', `SHA-256 ${image.sha256 ?? '未归档'} · ${image.observationId ?? '观测 ID 未归档'}`, 'evidence-hash'));
      figure.append(element('figcaption', `${index + 1}. ${image.label}`), link, metadata);
      figure.hidden = filter.value !== 'all' && image.frame !== filter.value;
      figures.push({ figure, frame: image.frame });
      grid.append(figure);
    }
    filter.addEventListener('change', () => { for (const item of figures) item.figure.hidden = filter.value !== 'all' && item.frame !== filter.value; });
    images.append(element('p', 'C 编号是 SAM3 临时分割候选，多个描述可以指向同一物体；不是永久 ID。R 标签定位清单实体。确认后的清单按永久 ID 合并，同一物体不会因提示词增加而重复建档。', 'muted'), filter, grid);
    const sam = evidence.segmentation.length ? section(`SAM3 提示与掩码 · ${evidence.segmentation.length} 次`) : element('div');
    for (const query of evidence.segmentation) {
      const item = element('details', '', 'sam-query');
      const proposals = query.proposals;
      item.append(element('summary', `相机 ${query.camera_id ?? '—'} · ${query.prompt ?? '提示未归档'} · ${Array.isArray(proposals) ? proposals.length : '未记录'} 个提议`),
        element('p', `${query.cache_hit ? '同帧缓存' : '分割调用'} · RGB SHA-256 ${query.rgb_sha256 ?? '未归档'}`, 'evidence-hash'));
      if (!Array.isArray(proposals)) item.append(element('p', '旧记录仅保存了查询元数据，没有掩码。', 'muted'));
      for (const [index, proposal] of (proposals ?? []).entries()) {
        const row = element('div', '', 'sam-proposal');
        row.append(element('p', `提议 ${index + 1} · score ${Number.isFinite(proposal.score) ? proposal.score.toFixed(4) : '未知'} · ${proposal.identity_status ?? '身份未记录'}`));
        const mask = element('button', '查看掩码', 'button');
        mask.addEventListener('click', () => { mask.disabled = true; void renderMask(runId, proposal, row, operation); });
        row.append(mask);
        if (proposal.points_world_m) {
          const cloud = element('button', '查看提议点云', 'button');
          cloud.addEventListener('click', async () => {
            cloud.disabled = true;
            await renderCloud(runId, proposal.points_world_m, [], row, operation);
          });
          row.append(cloud);
        }
        item.append(row);
      }
      sam.append(item);
    }
    const candidates = evidence.candidates.length ? section(`抓取候选与筛选 · ${evidence.candidates.length} 份`, true) : element('div');
    for (const replay of evidence.candidates) {
      const rows = candidateRows(replay);
      const item = element('details', '', 'candidate-evidence');
      const round = Number.isInteger(replay.sampling_round) && replay.sampling_round > 0
        ? ` · 采样第 ${replay.sampling_round} 轮` : '';
      item.append(element('summary', `${STAGE_NAMES[replay.stage] ?? replay.stage}${round} · ${replay.target_id ?? '目标 ID 未记录'} · ${rows.length} 个候选`));
      item.append(element('p', `执行臂：${replay.arm === 'A' ? '左臂 A' : replay.arm === 'B' ? '右臂 B' : '归档未记录'}。每个位姿显示完整 XYZ 工具坐标轴：X 红、Y 绿、Z 蓝；坐标方向来自归档矩阵，不额外生成候选。`, 'muted'));
      if (round) item.append(element('p', '候选编号保留本轮原始索引；不同轮次的同一编号表示不同候选。', 'muted'));
      item.append(element('p', replay.stage === 'raw_graspgen'
        ? replay.filter_all_generated
          ? `完整归档并预筛选全部 ${rows.length} 个生成位姿，筛后最多 ${replay.requested_candidate_count} 个进入 IK；此阶段还没有 IK 或轨迹判定。`
          : `完整归档 ${rows.length} 个生成位姿，本轮最多选择 ${replay.requested_candidate_count} 个进入预筛选；此阶段还没有 IK 或轨迹判定。`
        : `${rows.filter(row => row.status === 'passed').length} 通过 · ${rows.filter(row => row.status === 'failed').length} 未通过 · ${rows.filter(row => row.status === 'unknown').length} 未记录判定${replay.stage === 'before_ik' ? '（预筛选包含方向和已归档名义可达性，不代表 IK 或路径可行）' : ''}`, 'muted'));
      if (replay.stage === 'after_ik') item.append(element('p', replay.candidate_filters === undefined
        ? '后置筛选未归档：此处仅原始 IK 结果，最终可用性未记录。'
        : '此处显示已归档 IK 后筛选的最终可用性（eligible）；原始 IK 结果和筛选证据保留在详情。缺失、重复或损坏的筛选判定保持未知。', 'muted'));
      const controls = element('div', '', 'evidence-candidate-controls');
      const cloudSource = element('select'); cloudSource.setAttribute('aria-label', '点云证据来源');
      const cloudSources = candidateCloudSources(replay);
      for (const [index, source] of cloudSources.entries()) {
        const option = element('option', source.label); option.value = String(index); cloudSource.append(option);
      }
      cloudSource.value = '0';
      const sourceNote = element('p', cloudSources[0]?.description ?? '此阶段未归档可查看的点云。', 'evidence-source-note');
      if (replay.completion_provenance && Object.keys(replay.completion_provenance).length) {
        const provenance = replay.completion_provenance;
        const count = value => Number.isInteger(value) && value >= 0 ? value : '未归档';
        const distance = value => Number.isFinite(value) ? `${value} m` : '未归档';
        const details = element('details', '', 'completion-provenance');
        details.append(element('summary', '高度补全来源与点数（归档原文）'),
          element('p', `补全集：实测 ${count(provenance.measured_point_count)} 点 + 构造 ${count(provenance.synthetic_point_count)} 点；高度先验 ${distance(provenance.height_prior_m)}；支撑面 Z ${distance(provenance.support_z_m)}。`),
          element('pre', JSON.stringify(provenance, null, 2), 'candidate-detail'));
        item.append(details);
      } else if (replay.graspgen_input_source === 'height_completed') {
        item.append(element('p', '已标记使用高度补全，但补全来源细节未归档。', 'muted'));
      }
      if (replay.graspgen_input_sampling) {
        const sampling = replay.graspgen_input_sampling;
        const count = value => Number.isInteger(value) && value >= 0 ? value : '未归档';
        const spacing = Number.isFinite(sampling.spacing_m) ? `${sampling.spacing_m * 1000} mm` : '未归档';
        const details = element('details', '', 'graspgen-sampling-provenance');
        details.append(element('summary', 'GraspGen 专用表面采样（归档原文）'),
          element('p', `原补全集 ${count(sampling.source_geometry_point_count)} 点，另加 ${count(sampling.added_surface_point_count)} 个先验表面点，实际提交 ${count(sampling.actual_point_count)} 点。点间距 ${spacing}；这些新增点不是实测深度，也不改变地图体素。`),
          element('pre', JSON.stringify(sampling, null, 2), 'candidate-detail'));
        item.append(details);
      }
      const filter = element('select'); filter.setAttribute('aria-label', '候选结果筛选');
      for (const [value, label] of [['all', '全部候选'], ['passed', '通过'], ['failed', '未通过'], ['unknown', '未记录判定']]) {
        const option = element('option', label); option.value = value; filter.append(option);
      }
      const select = element('select'); select.setAttribute('aria-label', '选择候选编号');
      const draw = element('button', '打开 3D 点云与候选', 'button');
      draw.disabled = cloudSources.length === 0;
      const display = element('div');
      const details = element('pre', '', 'candidate-detail');
      let candidateViewer = null;
      let selectedRows = rows;
      cloudSource.addEventListener('change', () => {
        cloudOperations.set(display, ++cloudOperation);
        viewers.get(display)?.dispose();
        viewers.delete(display);
        candidateViewer = null;
        display.replaceChildren();
        sourceNote.textContent = `${cloudSources[Number(cloudSource.value)]?.description ?? '点云未归档'} 点击打开 3D 查看此来源。`;
      });
      const visibleRows = () => select.value === '' ? selectedRows
        : selectedRows.filter(row => String(row.index) === select.value);
      function refreshRows() {
        selectedRows = rows.filter(row => filter.value === 'all' || row.status === filter.value);
        select.replaceChildren();
        details.textContent = '';
        const all = element('option', `全部 ${selectedRows.length} 个`); all.value = ''; select.append(all);
        for (const row of selectedRows) {
          const eligibility = replay.stage === 'after_ik' ? ` · 最终可用 ${row.eligible === null ? '未知' : row.eligible ? '是' : '否'}` : '';
          const option = element('option', `#${row.index} · ${STATUS_NAMES[row.status]}${eligibility}${Number.isFinite(row.score) ? ` · score ${row.score.toFixed(4)}` : ''}`);
          option.value = String(row.index); select.append(option);
        }
      }
      filter.value = 'all'; refreshRows();
      filter.addEventListener('change', () => { refreshRows(); candidateViewer?.setCandidates(selectedRows); });
      select.addEventListener('change', () => {
        const chosen = selectedRows.filter(row => String(row.index) === select.value);
        candidateViewer?.setCandidates(select.value === '' ? selectedRows : chosen);
        details.textContent = chosen.length ? JSON.stringify(chosen[0], null, 2) : '';
      });
      draw.addEventListener('click', async () => {
        display.replaceChildren();
        const source = cloudSources[Number(cloudSource.value)];
        if (!source) return;
        sourceNote.textContent = source.description;
        const rendered = await renderCloud(runId, source.reference, visibleRows(), display, operation,
          source.transform, source.label);
        if (rendered) {
          candidateViewer = rendered;
          candidateViewer.setCandidates(visibleRows());
        }
      });
      item.append(element('p', '绿色为该阶段返回通过，红色为未通过，黄色为未知。工具线框仅标示候选位姿；后续碰撞、轨迹和执行验证另有记录。', 'muted'));
      controls.append(cloudSource, filter, select, draw);
      item.append(controls, sourceNote, display, details);
      candidates.append(item);
    }
    if (paths.length && onFile) {
      const files = section(`原始 JSON · ${paths.length} 个`);
      for (const path of paths) {
        const button = element('button', path, 'stage-button');
        button.addEventListener('click', () => onFile(path)); files.append(button);
      }
    }
  }
  return { show, clear, dispose: clear };
}
