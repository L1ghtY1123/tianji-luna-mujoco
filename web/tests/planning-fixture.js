export const planningSettings = {
  ik_position_threshold_m: .005, ik_rotation_threshold_rad: Math.PI / 36,
  ik_num_seeds: 32, trajopt_num_seeds: 4, approach_distance_m: .1,
  approach_axis: 'z', collision_check: true, time_scale: 1,
};

export const planningSchema = { properties: {
  ik_position_threshold_m: { type: 'number', title: 'IK 位置容差', description: '评估与规划使用相同 IK 容差。', group: 'shared', display_scale: 1000, unit: 'mm', step: 1, minimum: .001, maximum: .1 },
  ik_rotation_threshold_rad: { type: 'number', title: 'IK 角度容差', group: 'shared', display_scale: 180 / Math.PI, unit: '°', minimum: .001, maximum: 1 },
  ik_num_seeds: { type: 'integer', title: 'IK seeds', group: 'shared', minimum: 1, maximum: 512 },
  trajopt_num_seeds: { type: 'integer', title: 'TrajOpt seeds', group: 'planner', minimum: 1, maximum: 64 },
  approach_distance_m: { type: 'number', title: '接近距离', group: 'grasp', display_scale: 1000, unit: 'mm', minimum: .01, maximum: .5 },
  approach_axis: { type: 'string', title: '接近轴', group: 'grasp', enum: ['x', 'y', 'z'] },
  collision_check: { type: 'boolean', title: '碰撞检查', group: 'planner' },
  time_scale: { type: 'number', title: '轨迹时间倍率', description: '接近、抓取、提升三段统一应用。', group: 'execution', minimum: .1, maximum: 10 },
} };

export function planningPayload(settings = planningSettings, revision = 'settings-1') {
  return { settings: { ...settings }, schema: planningSchema, revision };
}
