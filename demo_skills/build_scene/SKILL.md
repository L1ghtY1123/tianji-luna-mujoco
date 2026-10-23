# build_scene

粉件和黑件均必须双臂在观测位姿，capture/generation/实际关节与观测一致。用全部当前清单实体的三RGBD建立全局2cm TSDF/ESDF；首次RANSAC桌面cuboid复用，不建立新桌面体素。选中目标层按抓取政策禁用，其他实体保留碰撞。只维护最新全局场景，禁止VLA位姿全局重建。

前提：{"observed": true, "inventory_finalized": true, "target_selected": true, "arm_selected": true, "destination_ready": true, "grasp_completed": false}。

输出：scene_ready, views_ready。
