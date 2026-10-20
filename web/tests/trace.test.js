import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import vm from 'node:vm';
import { formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver } from '../src/model.js';

const source = (await readFile(new URL('../src/trace.js', import.meta.url), 'utf8'))
  .replace(/^import .*;\r?\n/gm, '').replace('export function createTraceList', 'function createTraceList');

function render(events) {
  function node(tagName) {
    return { tagName, children: [], textContent: '', className: '',
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener() {},
    };
  }
  const context = vm.createContext({ document: { createElement: node }, formatJson, traceDetails, confidenceLabel, candidateCount, createEventTitleResolver });
  vm.runInContext(source, context);
  const container = node('div');
  const counter = node('span');
  context.createTraceList(container, counter)(events);
  function text(parent) { return [parent.textContent, ...parent.children.map(text)].join(' '); }
  return { container, counter, text: text(container), cardText: index => text(container.children[index]) };
}

test('mixed run records visibly distinguish JEV calls, skill execution, algorithms and system events in order', () => {
  const events = [
    { id: 11, kind: 'run_start', title: '开始任务' },
    { id: 12, kind: 'jev', title: '选择技能', data: { state: { facts: {} }, questions: { skill: {} }, output: { skill: 'plan_grasp' } } },
    { id: 13, kind: 'skill_start', title: 'plan_grasp' },
    { id: 14, kind: 'planning_batch_start', data: { batch_index: 1, batch_count: 1, candidate_indices: [7], arm: 'A' } },
    { id: 15, kind: 'execution_stage_complete', title: '执行提升轨迹', data: { skill: 'execute_grasp', stage: 'lift' } },
    { id: 16, kind: 'skill_complete', title: 'execute_grasp', data: { result: { completed: true } } },
    { id: 17, kind: 'run_end', title: '任务结束' },
  ];
  const view = render(events);
  assert.equal(view.counter.textContent, 7);
  const categories = ['系统事件', 'JEV 决策', 'skill 执行', '算法计算', 'skill 执行', 'skill 执行', '系统事件'];
  categories.forEach((category, index) => {
    const summary = view.container.children[index].children[0];
    const meta = summary.children[1].children.find(child => child.className === 'trace-meta');
    assert.equal(meta.children[0].textContent, category);
    assert.equal(summary.children[0].textContent, String(index + 1).padStart(2, '0'));
  });
  assert.match(view.cardText(1), /问题 · questions/);
  assert.match(view.cardText(1), /结构化输出 · output/);
  assert.match(view.cardText(4), /阶段完成.*execute_grasp 内部阶段.*不调用 JEV/);
  assert.match(view.cardText(5), /skill 完成/);
  assert.match(view.cardText(5), /"completed": true/);
});

test('only exact JEV event kinds receive decision labels and model-specific detail sections', () => {
  const view = render([
    { id: 1, kind: 'system_jev_status', title: '连接状态', data: { state: { connected: true }, questions: { diagnostic: 'raw' } } },
    { id: 2, kind: 'skill_error', title: '步骤失败', data: { error: 'invalid_scene', state: { preserved: true } } },
    { id: 3, kind: 'ik_batch_complete', data: { success_count: 1, duration_s: .25 } },
  ]);
  assert.match(view.cardText(0), /系统事件/);
  assert.match(view.cardText(0), /"connected": true/);
  assert.match(view.cardText(0), /"diagnostic": "raw"/);
  assert.doesNotMatch(view.cardText(0), /JEV 决策|问题 · questions|JEV 返回详情/);
  assert.match(view.cardText(1), /skill 执行.*skill 失败.*invalid_scene/);
  assert.match(view.cardText(1), /"preserved": true/);
  assert.doesNotMatch(view.cardText(1), /JEV 决策|问题 · questions/);
  assert.match(view.cardText(2), /算法计算/);
  assert.match(view.container.children[1].className, /trace-error/);
});

test('observation and place stages identify their owning skill', () => {
  for (const skill of ['move_to_observe', 'execute_place']) {
    const view = render([{ id: 1, kind: 'execution_stage_complete', title: '运动完成', data: { skill } }]);
    assert.match(view.text, new RegExp(`${skill} 内部阶段`));
    assert.match(view.text, new RegExp(`由 ${skill} skill 内部顺序执行`));
    assert.doesNotMatch(view.text, /execute_grasp/);
  }
});

test('a planning attempt without a solution is a system outcome returned to JEV, not a model call or terminal error', () => {
  const view = render([{ id: 1, kind: 'planning_attempt_failed', data: {
    skill: 'plan_grasp', arm: 'B', status: 'failed', code: 'planning_no_valid_candidate',
    error: 'no feasible lift', duration_s: 12.5, details: { failed_stages: ['lift'] },
  } }]);
  assert.match(view.text, /规划无解，交回 JEV/);
  assert.match(view.text, /系统事件/);
  assert.match(view.text, /Arm B.*尚未执行运动.*JEV 决定重新选臂或停止/);
  assert.match(view.text, /12\.50 s/);
  assert.match(view.text, /no feasible lift/);
  assert.match(view.text, /"failed_stages": \[/);
  assert.match(view.container.children[0].className, /trace-warning/);
  assert.doesNotMatch(view.container.children[0].className, /trace-error/);
  assert.doesNotMatch(view.text, /JEV 决策|问题 · questions|结构化输出 · output|自动换臂/);
});

test('batch trace shows parallel candidate IDs, arm, elapsed time and number of successful plans', () => {
  const view = render([
    { id: 1, kind: 'planning_batch_start', data: { batch_index: 1, batch_count: 2, candidate_indices: [7, 9, 12], arm: 'B' } },
    { id: 2, kind: 'planning_batch_complete', data: { batch_index: 1, batch_count: 2, candidate_indices: [7, 9, 12], arm: 'B', duration_s: 2.5, success_count: 2 } },
  ]);
  assert.match(view.text, /plan_grasp_batch 开始/);
  assert.match(view.text, /plan_grasp_batch 完成/);
  assert.match(view.text, /批次 1 \/ 2.*7, 9, 12.*Arm B/);
  assert.match(view.text, /2\.50 s/);
  assert.match(view.text, /成功 2 个/);
  assert.doesNotMatch(view.text, /继续尝试后续候选|上一候选失败/);
});

test('planning labels follow explicit stages and invocation ownership without rewriting evidence', () => {
  const events = [
    { id: 1, kind: 'skill_start', title: 'plan_place', invocation_id: 'place' },
    { id: 2, kind: 'skill_start', title: 'plan_grasp', invocation_id: 'grasp' },
    { id: 3, kind: 'planning_batch_start', title: 'plan_grasp_batch', invocation_id: 'batch', parent_invocation_id: 'place', data: { batch_index: 1 } },
    { id: 4, kind: 'planning_batch_complete', invocation_id: 'grasp', data: { success_count: 1 } },
    { id: 5, kind: 'planning_batch_complete', data: { planning_stage: 'place', success_count: 1 } },
    { id: 6, kind: 'planning_batch_start', title: 'legacy planner' },
    { id: 7, kind: 'planning_batch_complete', data: { batch_scope: 'generated_positions', release_reference: 'measured_floor' } },
    { id: 8, kind: 'planning_batch_start', title: 'cuRobo 放置偏航 batch 开始' },
  ];
  const original = structuredClone(events);
  const view = render(events);
  assert.match(view.cardText(2), /plan_place_batch 开始/);
  assert.match(view.cardText(3), /plan_grasp_batch 完成/);
  assert.match(view.cardText(4), /plan_place_batch 完成/);
  assert.match(view.cardText(5), /legacy planner/);
  assert.doesNotMatch(view.cardText(5), /plan_place_batch/);
  assert.match(view.cardText(6), /plan_place_batch 完成/);
  assert.match(view.cardText(7), /plan_place_batch 开始/);
  assert.deepEqual(events, original);
});

test('an unsuccessful batch is visibly marked without inventing a fallback or retry', () => {
  const view = render([{ id: 1, kind: 'planning_batch_complete', data: {
    batch_index: 1, batch_count: 1, candidate_indices: [7, 9], arm: 'A', success_count: 0,
    error: 'no feasible plans',
  } }]);
  assert.match(view.container.children[0].className, /trace-warning/);
  assert.match(view.text, /成功 0 个/);
  assert.match(view.text, /no feasible plans/);
  assert.doesNotMatch(view.text, /继续尝试后续候选/);
});

test('early planning success explains unplanned candidates in the visible summary without marking them failed', () => {
  const view = render([{ id: 1, kind: 'planning_batch_complete', data: {
    batch_index: 1, batch_count: 4, candidate_indices: [7, 9], arm: 'B', success_count: 1,
    stopped_early: true, unplanned_candidate_count: 48,
  } }]);
  const card = view.container.children[0];
  const notes = card.children[0].children[1].children
    .filter(child => child.tagName === 'p').map(child => child.textContent).join(' ');
  assert.match(notes, /批次 1 \/ 4/);
  assert.match(notes, /最高 score 可行轨迹.*停止后续批次.*48 个候选未规划/);
  assert.match(notes, /成功 1 个/);
  assert.doesNotMatch(notes, /失败/);
  assert.doesNotMatch(card.className, /trace-error|trace-warning/);
});

test('other batch events and unconfirmed early stops do not claim that planning stopped', () => {
  for (const [kind, stopped_early] of [
    ['planning_batch_start', true], ['ik_batch_complete', true],
    ['planning_batch_complete', false], ['planning_batch_complete', 'true'],
  ]) {
    const view = render([{ id: 1, kind, data: {
      batch_index: 1, batch_count: 4, candidate_indices: [7, 9], arm: 'B', success_count: 1,
      stopped_early, unplanned_candidate_count: 48,
    } }]);
    assert.doesNotMatch(view.text, /停止后续批次|个候选未规划/);
  }
});

test('planning renders every candidate ID when the batch includes sixteen original IDs', () => {
  const candidates = Array.from({ length: 16 }, (_, index) => index + 20);
  const view = render([{ id: 1, kind: 'planning_batch_start', data: {
    batch_index: 1, batch_count: 1, candidate_indices: candidates, arm: 'A',
  } }]);
  assert.ok(view.text.includes(`候选编号 ${candidates.join(', ')}`));
  assert.doesNotMatch(view.text, /仅展示/);
});

test('old summarized planning candidate IDs display the supplied prefix and original count', () => {
  const view = render([{ id: 1, kind: 'planning_batch_complete', data: {
    batch_index: 1, batch_count: 1, candidate_indices: { count: 16, first: [20, 21, 22] },
    arm: 'A', success_count: 2,
  } }]);
  assert.match(view.text, /候选编号 20, 21, 22.*仅展示 3 个，共 16 个/);
  assert.doesNotMatch(view.text, /20, 21, 22, 23/);
});

test('batched IK shows counts in summaries and details, including legacy summarized IDs', () => {
  const candidate_indices = { count: 100, first: [0, 1, 2, 3, 4, 5, 6, 7] };
  const view = render([
    { id: 1, kind: 'ik_batch_start', data: { batch_index: 1, batch_count: 2, candidate_indices, arm: 'B' } },
    { id: 2, kind: 'ik_batch_complete', data: { batch_index: 1, batch_count: 2, candidate_indices, arm: 'B', success_count: 12, duration_s: 3.5 } },
  ]);
  assert.match(view.text, /Batched IK 开始/);
  assert.match(view.text, /Batched IK 完成/);
  assert.match(view.text, /批次 1 \/ 2.*候选 100 个.*Arm B/);
  assert.match(view.text, /通过 12 个.*未通过 88 个/);
  assert.match(view.text, /"candidate_count": 100/);
  assert.doesNotMatch(view.text, /候选编号|candidate_indices|"first"|0, 1, 2/);
  assert.doesNotMatch(view.cardText(0), /通过|未通过/);
  assert.match(view.text, /3\.50 s/);
});

test('Batched IK ID lists are summarized throughout skill details without changing events or errors', () => {
  const candidate_indices = Array.from({ length: 100 }, (_, index) => index + 100);
  const batch = { batch_index: 2, batch_count: 2, candidate_indices, arm: 'A', success_count: 0,
    error: 'no valid IK', details: { candidate_index: 123, code: 'collision', position_error_m: 0.002 } };
  const events = [
    { id: 1, kind: 'ik_batch_start', data: batch },
    { id: 2, kind: 'ik_batch_complete', data: batch },
    { id: 3, kind: 'skill_complete', title: 'evaluate_arms', data: { result: {
      A: { tested_count: 200, valid_candidates: { count: 64, first: [{ candidate_index: 4, score: 0.98 }] }, batches: [batch] },
    } } },
    { id: 4, kind: 'skill_error', data: { skill: 'evaluate_arms', error: 'service disconnected', details: batch } },
  ];
  const original = structuredClone(events);
  const view = render(events);
  assert.doesNotMatch(view.text, /candidate_indices|候选编号/);
  assert.match(view.cardText(1), /通过 0 个.*未通过 100 个/);
  assert.match(view.container.children[1].className, /trace-warning/);
  for (const index of [1, 2, 3]) {
    assert.match(view.cardText(index), /"candidate_count": 100/);
    assert.match(view.cardText(index), /no valid IK/);
    assert.match(view.cardText(index), /"candidate_index": 123/);
    assert.match(view.cardText(index), /"position_error_m": 0.002/);
  }
  assert.match(view.cardText(3), /service disconnected/);
  assert.doesNotMatch(view.cardText(0), /通过|未通过/);
  assert.deepEqual(events, original);
});

test('raw JEV evidence keeps candidate IDs and missing IK counts do not invent failures', () => {
  const view = render([
    { id: 1, kind: 'jev', data: { state: { candidate_indices: [4, 9] }, questions: [{ options: ['A', 'B'] }] } },
    { id: 2, kind: 'ik_batch_complete', data: { batch_index: 1, arm: 'B', success_count: 2 } },
  ]);
  assert.match(view.cardText(0), /"candidate_indices": \[\s+4,\s+9/);
  assert.match(view.cardText(1), /候选 — 个.*Arm B/);
  assert.match(view.cardText(1), /通过 2 个/);
  assert.doesNotMatch(view.cardText(1), /未通过|NaN/);
});

test('placement events show partitions, generated yaws and the completed selection', () => {
  const view = render([
    { id: 1, kind: 'placement_layout', data: { target_count: 3, section_count: 4, yaw_count_per_position: 18 } },
    { id: 2, kind: 'placement_position_checked', data: { position_index: 0, position_world_m: [.4, -.1, .6],
      accepted_yaw_count: 18, generated_yaw_count: 18 } },
    { id: 3, kind: 'placement_candidate_selected', data: { position_index: 1, yaw_delta_deg: 20, candidate_index: 19 } },
  ]);
  assert.match(view.cardText(0), /3 个目标.*4 个分区.*18 个 yaw/);
  assert.match(view.cardText(1), /位置 1.*0.400.*-0.100.*0.600/);
  assert.match(view.cardText(1), /候选 18.*IK 和三维碰撞/);
  assert.match(view.cardText(2), /位置 2.*yaw 20°.*候选 19/);
  assert.match(view.cardText(2), /撤离.*停止后续候选/);
});
