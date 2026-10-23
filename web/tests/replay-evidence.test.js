import assert from 'node:assert/strict';
import test from 'node:test';

const evidence = await import('../src/replay-evidence.js').catch(() => ({}));

test('all generated poses remain distinct from the budgeted and prefiltered pool', () => {
  const poses = [0, 1, 2].map(index => ({ index, T_world_tcp: [[index]] }));
  const result = evidence.extractReplayEvidence({ candidate_replay: {
    stage: 'before_ik', all_generated_candidates: poses, requested_candidate_count: 2,
    candidates_before_ik: poses.slice(0, 2), pre_ik_eligible_indices: [1],
  } });
  assert.deepEqual(result.candidates.map(row => row.stage), ['raw_graspgen', 'before_ik']);
  const raw = evidence.candidateRows(result.candidates[0]);
  assert.equal(raw.length, 3);
  assert.ok(raw.every(row => row.status === 'unknown'));
  assert.match(raw[2].reason, /未送入/);
  assert.deepEqual(evidence.candidateRows(result.candidates[1]).map(row => row.status), ['failed', 'passed']);
  assert.equal(evidence.extractReplayEvidence({ candidate_replay: {
    stage: 'before_ik', candidates_before_ik: poses,
  } }).candidates.length, 1, 'historical runs cannot invent unarchived raw poses');
});

test('model inputs keep archived image order and refuse external image references', () => {
  assert.equal(typeof evidence.extractReplayEvidence, 'function');
  const result = evidence.extractReplayEvidence({ endpoint: '/v1/responses', request: { input: [{ content: [
    { type: 'input_text', text: '{"camera_id":"2","image_role":"current_rgb"}' },
    { type: 'input_image', image_url: { path: 'decisions/call-image-000.jpg', sha256: 'exact-hash' } },
    { type: 'input_image', image_url: { path: 'https://remote.test/image.jpg' } },
  ] }] } });
  assert.equal(result.images.length, 1);
  assert.equal(result.images[0].path, 'decisions/call-image-000.jpg');
  assert.equal(result.images[0].sha256, 'exact-hash');
  assert.match(result.images[0].label, /2/);
});

test('plain camera labels reset historical annotation context for current images', () => {
  const result = evidence.extractReplayEvidence({ endpoint: '/v1/decisions', request: { input: [{ content: [
    { type: 'input_text', text: '{"image_role":"initial_reference_annotations","camera":"2"}' },
    { type: 'input_image', image_url: { path: 'decisions/reference.jpg' } },
    { type: 'input_text', text: 'CURRENT camera 3 — original RGB' },
    { type: 'input_image', image_url: { path: 'decisions/current.jpg' } },
  ] }] } });
  assert.deepEqual(result.images.map(image => image.frame), ['reference', 'current']);
  assert.match(result.images[1].label, /相机 3.*当前.*原始 RGB/);
  assert.doesNotMatch(result.images[1].label, /初始/);
});

test('SAM3 empty results and unassociated proposals remain distinct evidence', () => {
  assert.equal(typeof evidence.extractReplayEvidence, 'function');
  const mask = { type: 'ndarray', path: 'artifacts/test.npz', key: 'array_0', shape: [2, 3] };
  const result = evidence.extractReplayEvidence({ result: { sam3_queries: [
    { camera_id: '2', prompt: 'pink object', rgb_sha256: 'rgb-hash', proposals: [] },
    { camera_id: '3', prompt: 'oval cylinder', proposals: [{ score: .91, mask, identity_status: 'unassociated_proposal' }] },
  ] } });
  assert.equal(result.segmentation.length, 2);
  assert.equal(result.segmentation[0].proposals.length, 0);
  assert.equal(result.segmentation[1].proposals[0].mask, mask);
});

test('IK response local indices map to original candidates without promoting missing outcomes', () => {
  assert.equal(typeof evidence.candidateRows, 'function');
  const identity = [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]];
  const replay = { stage: 'after_ik', candidate_indices: [51, 17, 92],
    request: { goal_matrices_world: [identity, identity, identity] },
    response: { items: [{ candidate_index: 0, success: false, status: 'unreachable' },
      { candidate_index: 1, success: true }] } };
  assert.deepEqual(evidence.candidateRows(replay).map(row => [row.index, row.status]),
    [[51, 'failed'], [17, 'passed'], [92, 'unknown']]);
});

test('IK eligibility joins original candidate indices and preserves raw results and filter evidence', () => {
  const goals = [[[1]], [[2]], [[3]]];
  const results = [{ candidate_index: 0, success: true }, { candidate_index: 1, success: true },
    { candidate_index: 2, success: false, error: 'unreachable' }];
  const filters = [{ candidate_index: 17, accepted: true, reason: 'accepted' },
    { candidate_index: 51, accepted: false, reason: 'predicted_gripper_collision',
      predicted_gripper_compatibility: { compatible: false, gap_m: .0003 } },
    { candidate_index: 92, accepted: true }];
  const rows = evidence.candidateRows({ stage: 'after_ik', candidate_indices: [51, 17, 92],
    request: { goal_matrices_world: goals }, response: { items: results }, candidate_filters: filters });
  assert.deepEqual(rows.map(row => [row.index, row.rawStatus, row.status, row.eligible]),
    [[51, 'passed', 'failed', false], [17, 'passed', 'passed', true], [92, 'failed', 'failed', false]]);
  assert.match(rows[0].reason, /predicted_gripper_collision/);
  assert.equal(rows[0].filterEvidence, filters[1]);
  assert.equal(rows[0].result, results[0]);
  assert.equal(rows[0].matrix, goals[0]);
});

test('historical IK without filters keeps raw status without claiming eligibility', () => {
  const [row] = evidence.candidateRows({ stage: 'after_ik', candidate_indices: [27],
    request: { goal_matrices_world: [[[1]]] }, response: { items: [{ candidate_index: 0, success: true }] } });
  assert.equal(row.status, 'passed');
  assert.equal(row.rawStatus, 'passed');
  assert.equal(row.eligible, null);
  assert.match(row.reason, /未归档.*仅原始 IK/);
});

for (const [label, filters, indices] of [
  ['duplicate', [{ candidate_index: 27, accepted: false }, { candidate_index: 27, accepted: true }], [27]],
  ['nonboolean acceptance', [{ candidate_index: 27, accepted: 'true' }], [27]],
  ['malformed list', { candidate_index: 27, accepted: true }, [27]],
  ['null list', null, [27]],
  ['null entry', [null, { candidate_index: 27, accepted: true }], [27]],
  ['string index', [{ candidate_index: '27', accepted: true }], [27]],
  ['missing candidate filter', [], [27]],
  ['unmatched original index', [{ candidate_index: 0, accepted: true }], [27]],
  ['missing original mapping', [{ candidate_index: 0, accepted: true }], undefined],
  ['duplicate original mapping', [{ candidate_index: 27, accepted: true }], [27, 27]],
]) {
  test(`IK ${label} filter evidence cannot promote raw success to eligible`, () => {
    const rows = evidence.candidateRows({ stage: 'after_ik', candidate_indices: indices,
      request: { goal_matrices_world: [[[1]], [[2]]] },
      response: { items: [{ candidate_index: 0, success: true }, { candidate_index: 1, success: true }] },
      candidate_filters: filters });
    assert.equal(rows[0].rawStatus, 'passed');
    assert.equal(rows[0].status, 'unknown');
    assert.equal(rows[0].eligible, null);
    assert.ok(rows[0].reason.length > 0);
  });
}

test('a positive filter cannot invent a missing raw IK response', () => {
  const [row] = evidence.candidateRows({ stage: 'after_ik', candidate_indices: [27],
    request: { goal_matrices_world: [[[1]]] }, response: { items: [] },
    candidate_filters: [{ candidate_index: 27, accepted: true }] });
  assert.equal(row.rawStatus, 'unknown');
  assert.equal(row.status, 'unknown');
  assert.equal(row.eligible, null);
});

test('failed IK stays failed when filters only cover raw-success candidates', () => {
  const [row] = evidence.candidateRows({ stage: 'after_ik', candidate_indices: [27],
    request: { goal_matrices_world: [[[1]]] }, response: { items: [{ candidate_index: 0, success: false }] },
    candidate_filters: [] });
  assert.equal(row.status, 'failed');
  assert.equal(row.rawStatus, 'failed');
  assert.equal(row.eligible, false);
});

test('candidate filters next to replay metadata survive extraction without replacing explicit nested evidence', () => {
  const filters = [{ candidate_index: 27, accepted: false }];
  const extracted = evidence.extractReplayEvidence({ candidate_indices: [27], candidate_filters: filters,
    candidate_replay: { stage: 'after_ik' } }).candidates[0];
  assert.equal(extracted.candidate_filters, filters);
  const malformed = evidence.extractReplayEvidence({ candidate_filters: filters,
    candidate_replay: { stage: 'after_ik', candidate_filters: null } }).candidates[0];
  assert.equal(malformed.candidate_filters, null);
});

test('after-plan status remains the raw planner result rather than applying IK filters', () => {
  const [row] = evidence.candidateRows({ stage: 'after_plan_grasp_batch', candidate_indices: [27],
    request: { goal_matrices_world: [[[1]]] }, response: { items: [{ candidate_index: 0, success: true }] },
    candidate_filters: [{ candidate_index: 27, accepted: false, reason: 'historical_ik_filter' }] });
  assert.equal(row.status, 'passed');
  assert.equal(row.eligible, undefined);
});

test('missing or malformed direction judgments remain unknown', () => {
  const rows = evidence.candidateRows({
    stage: 'before_ik',
    candidates_before_ik: [{ index: 1 }, { index: 2 }, { index: 3 }, { index: 4 }],
    direction_filter: { candidate_evidence: [
      { candidate_index: 1, tilt_passed: true },
      { candidate_index: 2, tilt_passed: 'true', away_from_robot_passed: true },
      { candidate_index: 3, tilt_passed: true, away_from_robot_passed: true },
      { candidate_index: 4, tilt_passed: false, away_from_robot_passed: true },
    ] },
  });
  assert.deepEqual(rows.map(row => row.status), ['unknown', 'unknown', 'passed', 'failed']);
  assert.match(rows[0].reason, /未记录/);
});

test('height-completed GraspGen choices use exact request arrays and keep original measurement separate', () => {
  assert.equal(typeof evidence.candidateCloudSources, 'function');
  const measured = [[.4, .2, .5]], completed = [...measured, [.4, .2, .4]];
  const actualObject = [[0, 0, 0], [0, 0, -.1]], actualScene = [[0, 0, 0], [.2, .3, -.2]];
  const frame = [[1,0,0,.4],[0,1,0,.2],[0,0,1,.5],[0,0,0,1]];
  const sources = evidence.candidateCloudSources({
    graspgen_input_source: 'height_completed', T_world_cloud: frame,
    measured_points_world_m: measured, completed_geometry_points_world_m: completed,
    graspgen_request: { scene_points_camera_m: actualScene, object_points_camera_m: actualObject },
  });
  assert.deepEqual(sources.map(source => source.id), ['graspgen-object', 'graspgen-scene', 'measured', 'completed']);
  assert.equal(sources[0].reference, actualObject);
  assert.equal(sources[0].transform, frame);
  assert.match(sources[0].label, /测量.*先验补全/);
  assert.match(sources[0].description, /不是全部实测/);
  assert.equal(sources[1].reference, actualScene);
  assert.doesNotMatch(sources[1].label, /补全/);
  assert.match(sources[1].description, /不应用于此场景数组/);
  assert.equal(sources[2].reference, measured);
  assert.equal(sources[2].transform, null);
  assert.equal(sources[3].reference, completed);
  assert.equal(sources[3].transform, null);
});

test('completion geometry cannot substitute for a missing actual GraspGen request', () => {
  const sources = evidence.candidateCloudSources({
    graspgen_input_source: 'height_completed', measured_points_world_m: [[0,0,1]],
    completed_geometry_points_world_m: [[0,0,1],[0,0,.9]],
  });
  assert.deepEqual(sources.map(source => source.id), ['measured', 'completed']);
  assert.ok(sources.every(source => !source.label.includes('GraspGen 实际')));
});

test('historical request composition stays unknown while explicit measured input stays measured', () => {
  const replay = {
    T_world_cloud: [[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]],
    graspgen_request: { object_points_camera_m: [[0,0,0]] },
    completed_geometry_points_world_m: [[0,0,0],[0,0,-.1]],
  };
  assert.match(evidence.candidateCloudSources(replay)[0].description, /构成未归档/);
  assert.match(evidence.candidateCloudSources({...replay, graspgen_input_source:'measured'})[0].label, /实测点/);
  assert.doesNotMatch(evidence.candidateCloudSources({...replay, graspgen_input_source:'measured'})[0].label, /补全/);
});

for (const stage of ['after_ik', 'after_plan_grasp_batch']) {
  test(`${stage} independently selects the same-round actual archived input before measured geometry`, () => {
    const ref = (key, count) => ({ type: 'ndarray', path: 'artifacts/round-3.npz', key, shape: [count, 3] });
    const actual = ref('actual_object', 2000), scene = ref('actual_scene', 5000);
    const replay = {
      stage, sampling_round: 3, candidate_indices: [93, 4],
      T_world_cloud: [[1,0,0,.3],[0,1,0,.1],[0,0,1,.6],[0,0,0,1]],
      graspgen_input_source: 'height_completed',
      graspgen_request: { object_points_camera_m: actual, scene_points_camera_m: scene },
      measured_points_world_m: ref('measured', 1200),
      completed_geometry_points_world_m: ref('completed_geometry', 1500),
    };
    const extracted = evidence.extractReplayEvidence({ candidate_replay: replay }).candidates[0];
    const sources = evidence.candidateCloudSources(extracted);
    assert.equal(extracted.sampling_round, 3);
    assert.deepEqual(extracted.candidate_indices, [93, 4]);
    assert.deepEqual(sources.map(source => source.id), ['graspgen-object', 'graspgen-scene', 'measured', 'completed']);
    assert.equal(sources[0].reference, actual);
    assert.equal(sources[1].reference, scene);
    assert.match(sources[0].description, /不是全部实测/);
  });
}
