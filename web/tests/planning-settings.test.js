import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { createPlanningSettingsEditor, displayPlanningValue, parsePlanningValue, planningFieldCategory, planningFieldMetadata } from '../src/planning-settings.js';
import { planningPayload, planningSchema, planningSettings } from './planning-fixture.js';

function editor() {
  function node(tagName) {
    return { tagName, children: [], listeners: {},
      append(...children) { this.children.push(...children); },
      replaceChildren(...children) { this.children = children; },
      addEventListener(type, listener) { this.listeners[type] = listener; },
    };
  }
  const container = node('div');
  const document = { createElement: node };
  const settings = createPlanningSettingsEditor(container, { document });
  function descendants(parent) { return parent.children.flatMap(child => [child, ...descendants(child)]); }
  return { settings, container,
    input(name) { return descendants(container).find(child => child.name === name); },
    change(name, value) { const input = this.input(name); assert.notEqual(input.disabled, true); input.value = value; input.listeners.input(); },
  };
}

test('planning lengths and angles display in mm/degrees and convert back to SI', () => {
  const distance = planningSchema.properties.ik_position_threshold_m;
  const angle = planningSchema.properties.ik_rotation_threshold_rad;
  assert.equal(displayPlanningValue(.005, distance), '5');
  assert.equal(displayPlanningValue(Math.PI / 36, angle), '5');
  assert.equal(parsePlanningValue('7', distance), .007);
  assert.ok(Math.abs(parsePlanningValue('10', angle) - Math.PI / 18) < 1e-12);
});

test('planning inputs reject empty, nonfinite, out-of-range, fractional integer and unsupported choices', () => {
  const field = planningSchema.properties.ik_position_threshold_m;
  for (const value of ['', ' ', 'NaN', 'Infinity', '0', '101']) assert.throws(() => parsePlanningValue(value, field));
  assert.throws(() => parsePlanningValue('2.5', planningSchema.properties.ik_num_seeds), /整数/);
  assert.throws(() => parsePlanningValue('q', planningSchema.properties.approach_axis));
  assert.equal(parsePlanningValue('false', planningSchema.properties.collision_check), false);
});

test('schema creates parameter categories with units, bounds and help and reads one shared SI draft', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  assert.equal(ui.container.children.length, 3);
  assert.equal(ui.input('ik_position_threshold_m').value, '5');
  assert.equal(ui.input('ik_position_threshold_m').step, '1');
  assert.equal(ui.input('ik_position_threshold_m').min, '1');
  assert.equal(ui.input('ik_position_threshold_m').max, '100');
  assert.equal(ui.settings.ready, true);
  assert.equal(ui.settings.dirty, false);
  ui.settings.setAdvancedLocked(false);
  ui.change('ik_position_threshold_m', '8');
  ui.change('time_scale', '2');
  assert.deepEqual(ui.settings.read(), { ...planningSettings, ik_position_threshold_m: .008, time_scale: 2 });
  assert.equal(ui.settings.dirty, true);
});

test('parameter categories keep calibration and self collision read-only while tool/world axes stay experimental', () => {
  assert.equal(planningFieldCategory('time_scale', { group: 'execution' }), 'common');
  assert.equal(planningFieldCategory('self_collision_check', { group: 'shared' }), 'system');
  assert.equal(planningFieldCategory('observation.A.0', { group: 'observation' }), 'system');
  assert.equal(planningFieldCategory('lift_in_tool_frame', { group: 'grasp' }), 'experimental');
  assert.equal(planningFieldCategory('place_retreat_axis', { group: 'place' }), 'experimental');
});

test('parameter metadata uses converted project defaults, schema ranges and verified project source names', () => {
  const metadata = planningFieldMetadata('ik_position_threshold_m', {
    ...planningSchema.properties.ik_position_threshold_m, default: .005,
  });
  assert.equal(metadata.defaultValue, '5 mm');
  assert.equal(metadata.range, '≥ 1，≤ 100 mm');
  assert.match(metadata.source, /planning_settings.py/);
  assert.match(metadata.impact, /IK/);
  assert.equal(planningFieldMetadata('unknown', { type: 'number' }).defaultValue, '未由 schema 提供');
});

test('read-only system values survive unrelated edits and cannot be changed through the editor', () => {
  const ui = editor();
  const payload = planningPayload();
  payload.schema = { properties: { ...payload.schema.properties,
    self_collision_check: { type: 'boolean', default: true, group: 'shared' },
  } };
  payload.settings.self_collision_check = false;
  ui.settings.load(payload);
  ui.settings.setAdvancedLocked(false);
  assert.equal(ui.input('self_collision_check').tagName, 'output');
  assert.equal(ui.input('self_collision_check').disabled, true);
  ui.input('self_collision_check').value = 'true';
  ui.change('time_scale', '2');
  assert.equal(ui.settings.read().self_collision_check, false);
});

test('advanced groups start locked while execution time remains editable', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  for (const name of ['ik_num_seeds', 'trajopt_num_seeds', 'approach_axis', 'collision_check']) {
    assert.equal(ui.input(name).disabled, true);
  }
  assert.equal(ui.input('time_scale').disabled, false);
  ui.change('time_scale', '2');
  assert.deepEqual(ui.settings.read(), { ...planningSettings, time_scale: 2 });
});

test('locking advanced fields preserves unapplied values, dirty state and revision', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  ui.settings.setAdvancedLocked(false);
  assert.equal(ui.input('ik_num_seeds').disabled, false);
  ui.change('ik_num_seeds', '16');
  ui.settings.setAdvancedLocked(true);
  assert.equal(ui.input('ik_num_seeds').disabled, true);
  assert.equal(ui.input('ik_num_seeds').value, '16');
  assert.equal(ui.settings.dirty, true);
  assert.equal(ui.settings.revision, 'settings-1');
  assert.equal(ui.settings.read().ik_num_seeds, 16);
  ui.settings.setAdvancedLocked(false);
  assert.equal(ui.input('ik_num_seeds').value, '16');
});

test('server refreshes retain the advanced lock state and a new editor starts locked', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  ui.settings.sync(planningPayload({ ...planningSettings, ik_num_seeds: 16 }, 'settings-2'));
  assert.equal(ui.input('ik_num_seeds').disabled, true);
  ui.settings.setAdvancedLocked(false);
  ui.settings.accept(planningPayload());
  assert.equal(ui.input('ik_num_seeds').disabled, false);
  const reopened = editor();
  reopened.settings.load(planningPayload());
  assert.equal(reopened.input('ik_num_seeds').disabled, true);
});

test('invalid field remains editable but cannot create an API payload', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  ui.change('time_scale', '');
  assert.equal(ui.settings.dirty, true);
  assert.throws(() => ui.settings.read(), /轨迹时间倍率.*数值/);
});

test('new server revision updates clean controls but preserves a dirty draft and records conflict', () => {
  const ui = editor();
  ui.settings.load(planningPayload());
  ui.settings.sync({ settings: { ...planningSettings, time_scale: 2 }, revision: 'settings-2' });
  assert.equal(ui.input('time_scale').value, '2');
  ui.change('time_scale', '3');
  ui.settings.sync({ settings: { ...planningSettings, time_scale: 4 }, revision: 'settings-3' });
  assert.equal(ui.input('time_scale').value, '3');
  assert.equal(ui.settings.conflict, true);
  assert.equal(ui.settings.revision, 'settings-2');
  ui.settings.accept(planningPayload({ ...planningSettings, time_scale: 4 }, 'settings-3'));
  assert.equal(ui.input('time_scale').value, '4');
  assert.equal(ui.settings.dirty, false);
  assert.equal(ui.settings.conflict, false);
});

test('an initial schema fetch cannot overwrite a newer state snapshot', () => {
  const ui = editor();
  ui.settings.sync({ settings: { ...planningSettings, time_scale: 4 }, revision: 'settings-2' });
  assert.equal(ui.settings.ready, false);
  ui.settings.load(planningPayload());
  assert.equal(ui.settings.revision, 'settings-2');
  assert.equal(ui.input('time_scale').value, '4');
});

test('editing one parameter preserves the exact SI values of untouched angle inputs', () => {
  const ui = editor();
  const settings = { ...planningSettings, ik_rotation_threshold_rad: .05 };
  ui.settings.load(planningPayload(settings));
  ui.change('time_scale', '2');
  assert.equal(ui.settings.read().ik_rotation_threshold_rad, .05);
});

test('the separate place group starts locked, converts millimeters and preserves grasp geometry', () => {
  const ui = editor();
  const payload = planningPayload();
  payload.schema = { properties: { ...payload.schema.properties,
    place_approach_offset_m: { type: 'number', title: 'Place approach 偏移', group: 'place', display_scale: 1000, unit: 'mm', minimum: -.5, maximum: .5 },
    place_approach_in_tool_frame: { type: 'boolean', title: 'Place tool frame', group: 'place' },
    dwell_release_s: { type: 'number', title: '释放后稳定等待', group: 'execution', minimum: 0, maximum: 30 },
  } };
  payload.settings = { ...payload.settings, place_approach_offset_m: .1, place_approach_in_tool_frame: false, dwell_release_s: 1 };
  ui.settings.load(payload);
  assert.equal(ui.input('place_approach_offset_m')?.value, '100');
  assert.equal(ui.input('place_approach_offset_m').disabled, true);
  assert.equal(ui.input('dwell_release_s').disabled, false);
  ui.settings.setAdvancedLocked(false);
  ui.change('place_approach_offset_m', '150');
  assert.equal(ui.settings.read().place_approach_offset_m, .15);
  assert.equal(ui.settings.read().approach_distance_m, .1);
});

test('visible placement height is ten millimeters and optional nested measured-point filters roundtrip', () => {
  const ui = editor();
  const payload = planningPayload();
  payload.schema = { properties: { ...payload.schema.properties,
    place_release_height_m: { type: 'number', title: '物体离盒底高度', description: '不叠加碰撞间距。', group: 'place', display_scale: 1000, unit: 'mm', minimum: 0, maximum: .1 },
    pointcloud_filter: { $ref: '#/$defs/PointcloudFilterSettings' },
  }, $defs: { PointcloudFilterSettings: { properties: {
    enabled: { type: 'boolean', title: '启用可见点云过滤', group: 'filter' },
    radius_m: { type: 'number', title: '半径过滤邻域', group: 'filter', display_scale: 1000, unit: 'mm', exclusiveMinimum: 0, maximum: .5 },
    dbscan_preview: { type: 'boolean', title: 'DBSCAN 点簇诊断', group: 'filter' },
  } } } };
  payload.settings = { ...payload.settings, place_release_height_m: .01,
    pointcloud_filter: { enabled: false, radius_m: .01, dbscan_preview: false } };
  ui.settings.load(payload);
  assert.equal(ui.input('place_release_height_m').value, '10');
  assert.equal(ui.input('pointcloud_filter.enabled').value, 'false');
  assert.equal(ui.input('pointcloud_filter.enabled').disabled, true);
  assert.deepEqual(ui.settings.read(), payload.settings);
  ui.settings.setAdvancedLocked(false);
  ui.change('pointcloud_filter.radius_m', '15');
  ui.change('pointcloud_filter.enabled', 'true');
  assert.equal(ui.input('place_release_height_m').disabled, true);
  const values = ui.settings.read();
  assert.deepEqual(values.pointcloud_filter, { enabled: true, radius_m: .015, dbscan_preview: false });
  assert.equal(values.place_release_height_m, .01);
  assert.equal(values.approach_axis, payload.settings.approach_axis);
  assert.equal(values.approach_distance_m, payload.settings.approach_distance_m);
  assert.equal(Object.keys(values).some(name => name.includes('.')), false);
});

test('observation joint arrays remain read-only while attachment fields unlock and preserve exact SI snapshots', () => {
  const ui = editor();
  const payload = planningPayload();
  const joint = index => ({ type: 'number', title: `A 关节 ${index + 1}`, group: 'observation',
    unit: '°', display_scale: 180 / Math.PI, minimum: -2, maximum: 2, step: .1 });
  payload.schema = { properties: { ...payload.schema.properties,
    observation: { $ref: '#/$defs/ObservationSettings' }, attachment: { $ref: '#/$defs/AttachmentSettings' },
  }, $defs: {
    ObservationSettings: { properties: { A: { type: 'array', prefixItems: Array.from({ length: 7 }, (_, i) => joint(i)) },
      B: { type: 'array', prefixItems: Array.from({ length: 7 }, (_, i) => ({ ...joint(i), title: `B 关节 ${i + 1}` })) } } },
    AttachmentSettings: { properties: {
      voxel_size_m: { type: 'number', group: 'attachment', display_scale: 1000, minimum: .001, maximum: .05 },
      margin_m: { type: 'number', group: 'attachment', display_scale: 1000, minimum: 0, maximum: .03 },
      sphere_budget: { type: 'integer', group: 'attachment', minimum: 1, maximum: 128 },
    } },
  } };
  const joints = [.123456789123, 0, 0, 0, 0, 0, 0];
  payload.settings = { ...payload.settings, observation: { A: [...joints], B: [...joints] },
    attachment: { voxel_size_m: .01, margin_m: .001, sphere_budget: 128 } };
  const original = structuredClone(payload.settings);
  ui.settings.load(payload);
  assert.equal(ui.input('observation.A.0').disabled, true);
  assert.equal(ui.input('attachment.voxel_size_m').value, '10');
  assert.equal(ui.input('attachment.margin_m').disabled, true);
  assert.deepEqual(ui.settings.read(), payload.settings);
  ui.settings.setAdvancedLocked(false);
  ui.change('attachment.sphere_budget', '129');
  assert.throws(() => ui.settings.read(), /128/);
  ui.change('attachment.sphere_budget', '128');
  assert.equal(ui.input('observation.B.6').disabled, true);
  assert.equal(ui.input('observation.B.6').tagName, 'output');
  ui.change('attachment.margin_m', '2');
  ui.settings.setAdvancedLocked(true);
  const draft = ui.settings.read();
  assert.equal(draft.observation.B[6], joints[6]);
  assert.equal(draft.observation.A[0], joints[0]);
  assert.equal(draft.attachment.margin_m, .002);
  assert.deepEqual(payload.settings, original);
  assert.equal(ui.input('observation.B.6').disabled, true);
});

test('untouched actual URDF joint limits survive rounded degree display when changing dwell time', async () => {
  const urdf = await readFile(new URL('../../assets/marvin_urdf/marvin_with_grippers.urdf', import.meta.url), 'utf8');
  const limits = Object.fromEntries([['A', 'L'], ['B', 'R']].map(([arm, suffix]) => [arm,
    Array.from({ length: 7 }, (_, index) => {
      const joint = urdf.match(new RegExp(`<joint name="Joint${index + 1}_${suffix}"[^>]*>[\\s\\S]*?</joint>`))[0];
      return { type: 'number', group: 'observation', title: `${arm} joint ${index + 1}`,
        display_scale: 180 / Math.PI, unit: '°',
        minimum: Number(joint.match(/<limit[^>]*lower="([^"]+)"/)[1]),
        maximum: Number(joint.match(/<limit[^>]*upper="([^"]+)"/)[1]) };
    })]));
  // This shipped limit rounds outward in degrees; parsing the display is not lossless.
  assert.throws(() => parsePlanningValue(displayPlanningValue(limits.A[0].maximum, limits.A[0]), limits.A[0]));
  for (const bound of ['minimum', 'maximum']) {
    const ui = editor();
    const payload = planningPayload();
    payload.schema = { properties: { ...payload.schema.properties,
      observation: { properties: Object.fromEntries(Object.entries(limits).map(([arm, fields]) =>
        [arm, { type: 'array', prefixItems: fields }])) },
      dwell_release_s: { type: 'number', title: '释放后稳定等待', group: 'execution', minimum: 0, maximum: 30 },
    } };
    payload.settings = { ...payload.settings, dwell_release_s: 1,
      observation: Object.fromEntries(Object.entries(limits).map(([arm, fields]) => [arm, fields.map(field => field[bound])])) };
    ui.settings.load(payload);
    assert.equal(ui.input('observation.A.0').disabled, true);
    ui.change('dwell_release_s', '2');
    assert.deepEqual(ui.settings.read(), { ...payload.settings, dwell_release_s: 2 });
    ui.settings.setAdvancedLocked(false);
    assert.equal(ui.input('observation.A.0').disabled, true);
    ui.input('observation.A.0').value = String((limits.A[0].maximum + .001) * 180 / Math.PI);
    assert.deepEqual(ui.settings.read(), { ...payload.settings, dwell_release_s: 2 });
  }
});
