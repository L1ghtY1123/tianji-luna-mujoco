// Display names come only from archived objects. They never rebind an instance.
export const SKILL_LABELS = {
  capture_scene: '采集同步 RGBD', ground_task: 'RGBD 任务定位与身份登记',
  observe_scene: '三相机观测', select_target: '选择待处理物体',
  identify_target: '绑定当前物体', identify_destination: '确定目标区域',
  choose_arm: '选择执行臂', build_scene: '更新空间记忆',
  generate_grasps: '生成抓取位姿', plan_grasp: '规划抓取',
  execute_grasp: '执行抓取', verify_grasp: '核验夹持',
  perceive_placement_zone: '识别放置区域',
  plan_place: '规划放置', execute_place: '放置与撤离', verify_place: '核验放置',
  prepare_unpack: '准备拆包', vla_unpack: '执行 VLA 拆包', verify_unpack: '核验拆包',
  move_to_observe: '调整观测位姿', return_to_initial_pose: '返回初始位姿',
  verify_task_completion: '核验任务完成',
  prepare_failure_return: '终止失败：停止并准备回位', observe_after_failure: '终止失败：当前位置补观测',
  build_failure_scene: '终止失败：更新回位地图', return_after_failure: '终止失败：回观测位姿并停止',
};

export function archivedEntityLabels(events = []) {
  const labels = new Map();
  for (const event of events) {
    const progress = event.data?.task_progress;
    if (!progress) continue;
    for (const records of [progress.scene_entities, progress.entities, progress.objects]) {
      for (const entity of Array.isArray(records) ? records : Object.values(records ?? {})) {
        const label = entity.description || entity.name;
        if (entity.id && label && !labels.has(entity.id)) labels.set(entity.id, label);
      }
    }
  }
  return labels;
}

export function displayEntityName(id, labels, fallback = '未命名物体') {
  const label = labels instanceof Map ? labels.get(id) : labels?.[id];
  return typeof label === 'string' && label.trim() ? label.trim().replace(/[。.]$/, '') : fallback;
}
