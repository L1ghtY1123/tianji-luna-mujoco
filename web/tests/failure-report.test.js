import assert from 'node:assert/strict';
import test from 'node:test';
import { failureSummary, renderFailureReport, loadArchivedFailureReport } from '../src/failure-report.js';

function container() {
  const document = { createElement: tag => ({ tag, ownerDocument: document, children: [], textContent: '',
    append(...children) { this.children.push(...children); },
    replaceChildren(...children) { this.children = children; },
  }) };
  return document.createElement('section');
}
const text = node => [node.textContent, ...node.children.map(text)].join(' ');

test('multimodal questions show their actual text rather than object coercions', () => {
  const target = container();
  renderFailureReport(target, { status: 'failed', failure_report: { summary: 'placement uncertain', checks: [
    { name: 'place_result', question: [
      { type: 'input_text', text: 'Is the pink object on the table?' },
      { type: 'input_text', text: 'CURRENT RGB camera 3' },
      { type: 'input_image', image_url: { path: 'decisions/current.jpg' } },
    ], actual: .5, threshold: .8 },
  ] } });
  assert.match(text(target), /pink object.*CURRENT RGB camera 3/s);
  assert.doesNotMatch(text(target), /\[object Object\]/);
});

test('compacted archive checks render and hydrate from the original event without rewriting evidence', async () => {
  const checks = Array.from({length: 12}, (_, index) => ({name: `check-${index}`, actual: index}));
  const events = [{kind:'run_end', artifact:'artifacts/event-00099.json', data:{failure_report:{
    summary:'coverage rejected', checks:{count:12,first:checks.slice(0,3)}}}}];
  const before=JSON.stringify(events);
  const target=container();
  renderFailureReport(target, {}, events);
  assert.match(text(target), /3.*12/);
  const hydrated=await loadArchivedFailureReport(events, async path=>{
    assert.equal(path,'artifacts/event-00099.json');
    return {failure_report:{summary:'coverage rejected',checks}};
  });
  renderFailureReport(target, {}, hydrated);
  assert.match(text(target), /check-11/);
  assert.equal(JSON.stringify(events),before);
  const unavailable=await loadArchivedFailureReport(events, async()=>{throw Error('offline')});
  renderFailureReport(target, {}, unavailable);
  assert.match(text(target), /offline/);
  assert.match(text(target), /check-0/);
});

test('failure report shows recorded comparisons, provenance and unknowns without changing evidence', () => {
  const run = { status: 'failed', failure_report: { component: 'Decisions', code: 'identity_unknown',
    summary: '身份判断未达到阈值', stage: 'observe_scene', target_id: 'part', arm: 'A',
    observation_id: 'frame-1', request_hash: 'hash-1', checks: [
      { name: 'identity', question: 'Is this the selected object?', actual: 0, threshold: .8,
        comparison: '>=', camera_id: '3', candidate_id: 'candidate-1', entity_id: 'part' },
    ], known_facts: ['已收到三相机画面'], unknowns: ['未确认目标身份'],
    evidence_refs: [{ event_id: 12, artifact: 'decisions/12.json', request_hash: 'hash-1' }], next_step: '核对本次观测与身份问题' } };
  const before = JSON.stringify(run);
  const target = container();
  renderFailureReport(target, run);
  assert.equal(failureSummary(run), '身份判断未达到阈值');
  assert.equal(target.children[0].textContent, '身份判断未达到阈值');
  assert.equal(target.children[1].tag, 'details');
  assert.match(text(target), /实际值：0.*要求：>= 0.8/s);
  for (const evidence of ['Decisions', 'observe_scene', 'part', 'frame-1', 'hash-1', 'candidate-1',
    'Is this the selected object?', '未确认目标身份', 'decisions/12.json', '核对本次观测与身份问题']) {
    assert.ok(text(target).includes(evidence), evidence);
  }
  assert.equal(JSON.stringify(run), before);
});

test('missing fields remain unrecorded and repeated rendering preserves expanded evidence', () => {
  const target = container();
  const run = { failure_report: { code: 'unknown', checks: [{ name: 'depth', actual: false }] } };
  renderFailureReport(target, run);
  target.children[1].open = true;
  renderFailureReport(target, run);
  assert.equal(target.children[1].open, true);
  assert.match(text(target), /实际值：false.*阈值未记录/s);
  assert.doesNotMatch(text(target), /SAM3|>=|已通过/);
  renderFailureReport(target, {});
  assert.equal(target.hidden, true);
});

test('legacy errors show original recorded details without inventing segmentation diagnoses', () => {
  const target = container();
  const events = [{ id: 7, kind: 'task_failure', data: { code: 'visual_identity_unknown',
    stage: 'observe_scene', evidence: { error_details: { missing_ids: ['part'], probabilities: { part: .4 } } } } }];
  renderFailureReport(target, { status: 'failed', error_details: { request_hash: 'hash-old' } }, events);
  assert.match(text(target), /visual_identity_unknown/);
  assert.match(text(target), /hash-old/);
  assert.match(text(target), /missing_ids/);
  assert.doesNotMatch(text(target), /SAM3|分割失败|空掩码/);
});

test('replay reads the archived report and falls back to raw skill errors', () => {
  const target = container();
  renderFailureReport(target, {}, [{ id: 9, kind: 'run_end', data: {
    failure_report: { summary: '路径范围覆盖不足', unknowns: ['区域 x 未观测'] } } }]);
  assert.match(text(target), /路径范围覆盖不足.*区域 x 未观测/s);
  renderFailureReport(target, {}, [{ id: 10, kind: 'skill_error', data: { error: 'custom_failure', details: { actual: 12 } } }]);
  assert.match(text(target), /custom_failure.*actual/s);
});

test('planning failures retain structured measurements and count failures per stage', () => {
  const target = container();
  renderFailureReport(target, {}, [{ id: 3, kind: 'skill_error', data: { failure_report: {
    summary: '覆盖证据不足', checks: [{ name: 'coverage', actual: { center: [0, .2, .3], radius: .01, sample_index: 0 } }],
    failure_counts: [{ stage: 'approach', code: 'coverage_missing', count: 4 },
      { stage: 'lift', code: 'collision', count: 2 }],
  } } }]);
  assert.match(text(target), /center：\[0,0.2,0.3\].*轨迹采样索引：0/s);
  assert.match(text(target), /按阶段记录的失败次数.*approach.*4 次.*lift.*2 次/s);
  assert.doesNotMatch(text(target), /候选总数|6 个候选/);
});

test('a successful terminal result does not present a recovered failure as the run conclusion', () => {
  const target = container();
  const events = [{ kind: 'skill_error', data: { failure_report: { summary: 'Earlier attempt failed' } } },
    { kind: 'run_end', data: { status: 'succeeded', failure_report: null } }];
  renderFailureReport(target, { status: 'succeeded' }, events);
  assert.equal(target.hidden, true);
  assert.equal(failureSummary({ status: 'succeeded' }, events), '');
});

test('successful experiments label recording gaps as archive issues instead of task failure', () => {
  const target = container();
  renderFailureReport(target, { status: 'succeeded' }, [], { additionalErrors: ['scene capture gap of 1.36s exceeded 1.0s'] });
  assert.equal(target.hidden, false);
  assert.match(text(target), /任务已成功.*归档.*scene capture gap/s);
  assert.doesNotMatch(text(target), /失败原因未记录/);
});

test('SAM3 zero candidates and identity answers retain distinct evidence without invented score failures', () => {
  const target = container();
  renderFailureReport(target, { failure_report: { summary: '当前观测未确认目标', checks: [
    { name: 'sam3_query', prompt: 'small object', candidate_count: 0, candidate_scores: [],
      status: 'no_matches', reason: 'No matching mask returned', camera_id: '2', query_id: 'query-4' },
    { name: 'identity_candidate-1', question: 'Does this candidate match the target?',
      answer: { choice: 'unknown', confidence: .61 }, actual: .61, threshold: .8, comparison: '>=',
      metric: 'confidence', sam3_score: .9, expected: 'part-1', candidate_id: 'candidate-1' },
  ] } });
  const value = text(target);
  assert.match(value, /查询描述：small object.*候选数量：0.*候选分数：\[\].*记录状态：no_matches.*记录原因：No matching mask returned/s);
  assert.match(value, /query-4/);
  assert.match(value, /实际值：0.61.*要求：>= 0.8.*原始回答：\{"choice":"unknown","confidence":0.61\}.*指标：confidence.*SAM3 分数：0.9.*预期值：part-1/s);
  assert.doesNotMatch(value, /分割低分|SAM3 分数不足|SAM3.*0.8/);
});

test('coverage measurements use separate labeled rows and an expected condition instead of a numeric threshold', () => {
  const target = container();
  const actual = { covered: false, first_unobserved_sample_index: 0,
    first_unobserved_sphere_world_m: [.1, .2, .3, .04], maximum_motion_envelope_m: .005,
    minimum_capture_containment_gap_m: .002, extra_measurement: { source: 'depth' } };
  const before = JSON.stringify(actual);
  renderFailureReport(target, { failure_report: { summary: '覆盖证据不足',
    checks: [{ name: 'coverage', actual, expected: 'covered=true' }] } });
  const check = target.children[1].children.find(child => child.tag === 'section');
  const rows = check.children.find(child => child.tag === 'ul').children;
  assert.equal(rows.length, 6);
  assert.equal(rows[0].textContent, '覆盖检查通过：false');
  assert.match(text(rows[2]), /首个未覆盖碰撞球（世界坐标 x,y,z 与半径，m）：\[0.1,0.2,0.3,0.04\]/);
  assert.match(text(check), /最大运动包络（m）：0.005/);
  assert.match(text(check), /距采集时机器人包络的最小包含缺口（m）：0.002/);
  assert.match(text(check), /extra_measurement：.*source：depth/s);
  assert.match(text(check), /按预期条件检查.*预期值：covered=true/s);
  assert.doesNotMatch(text(check), /阈值未记录|"covered":false/);
  assert.equal(JSON.stringify(actual), before);
});

test('known missing-diagnostic tags are readable while unknown text and lookup codes remain original', () => {
  const target = container();
  renderFailureReport(target, { failure_report: { component: 'sam3', code: 'identity_unknown', unknowns: [
    'sam3_internal_filter_threshold', 'sam3_internal_rejection_reason',
    'model_internal_reason_not_returned', 'sam3_score_not_recorded', 'new_unknown_tag',
  ] } });
  const value = text(target);
  assert.match(value, /组件：sam3.*错误码：identity_unknown/s);
  for (const label of ['SAM3 服务未返回内部过滤阈值', 'SAM3 服务未返回内部拒绝原因',
    '接口未返回模型内部判断理由', '此候选未记录 SAM3 分数', 'new_unknown_tag']) assert.ok(value.includes(label));
});
