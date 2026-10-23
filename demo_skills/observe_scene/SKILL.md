# observe_scene

双臂必须在固定观测位姿。首次用三RGB联合Responses理解指令并建立固定ID清单和技能程序；后续用同次三RGBD、SAM3及一次联合Decisions关联当前清单实体。VLA后复用verify_unpack回位后的确认capture；全部抓取前均须获得当前全局观测，不允许在VLA位姿直接抓取。多个SAM描述和临时C标签不会生成多个永久ID。此技能仅感知，不发布碰撞图。

前提：{"grasp_held": false, "grasp_view_ready": true}。

输出：observed, inventory_ready。
