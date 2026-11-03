# 相机内外参参考核查（2026-10-27）

问题在标定资源的设备身份混用：三路外参已经更新为 D405 / Gemini 335Lg 的实测结果，内参仍来自 2026-09 的芯明 VirtualCamera。MuJoCo 渲染适配器本来就在使用完整内参，没有只保留 fovy。本次仅修正资源及相机回归，未修改相机代码、MuJoCo SDK 或参考项目。

## 来源与实际配置

- 用户指定参考：`C:/project/main/tianji_base`，全程只读。
- `extensions/cameras/orbbec-camera/cameras.json`：右腕 RealSense D405 SN `260522273287`、左腕 D405 SN `260422273023`、头部 Orbbec Gemini 335Lg SN `CPB94630009P`，三路均为 640×480、60 fps。
- `extensions/cameras/camera-extrinsics/defaults/camera_extrinsics.json`：上述相同 SN 的 2026-10-20 发布结果。
- 本项目 `resources/cameras/user-extrinsics/camera_{right_wrist,left_wrist,head}_extrinsic.json`：分别保留 60、45、18 份原始样本。每个文件的全部 `samples[*].camera_calibration` 完全一致，SN 与参考绑定、外参 SN 相同；其完整精度 K 和分辨率作为本次内参来源。
- `xinming-camera/driver/camera_models.json` 位于参考项目 `extensions/cameras/` 下，记录的是旧芯明设备（腕 SN `411701X262200017` / `411701X262200004`、头 SN `BBA03230495`），不能作为 D405/Orbbec 内参来源。
- 只读查询远端已有 `jodell-real-singleton` 的 `/api/core/v1/capabilities`，实际 `jodell.camera.stream/v1` 和 exposure Provider 均为 `jodell.camera.orbbec`，`selection_source=profile`。读取配置没有发起相机启动、图像采集或机器人动作。

远端该其他用户实例的 `var/camera-extrinsics/camera_extrinsics.json` 另有旧运行期覆盖：腕部保存于 10-15，头部保存于 08-26，缺少 SN，且不等于参考默认值。参考插件允许运行期文件覆盖默认值。本次严格采用用户指定源码默认值及匹配 SN 的原始样本；未修改其他实例，也不宣称其在线标定已经同步。

## 内参与视野差异

K 的顺序为 `(fx, fy, cx, cy)`，单位为像素；矩阵为 `[[fx,0,cx],[0,fy,cy],[0,0,1]]`。

| 相机 | 原仿真 K | 对应实测 K | 分辨率变化 |
|---|---|---|---|
| 1 右腕 B | (374.025974025974, 374.025974025974, 317.9415584415584, 235.86363636363635) | (394.2907409667969, 393.8032531738281, 318.0513610839844, 229.73512268066406) | 640×480 → 640×480 |
| 2 左腕 A | (369.2307692307693, 369.2307692307693, 318.9871794871795, 243.08974358974362) | (393.44769287109375, 392.82000732421875, 320.0536193847656, 237.30731201171875) | 640×480 → 640×480 |
| 3 头部 | (291.8918918918919, 291.8918918918919, 320.0405405405405, 194.6351351351351) | (370.1304931640625, 370.15179443359375, 321.00567626953125, 241.04464721679688) | 640×360 → 640×480 |

按整数像素中心、图像边缘 `-0.5` 与 `W-0.5` 计算非对称针孔视角：

`HFOV = atan((cx+0.5)/fx) + atan((W-0.5-cx)/fx)`；VFOV 同理。

| 相机 | 原 HFOV × VFOV | 修正 HFOV × VFOV | fx / fy 增幅 |
|---|---|---|---|
| 右腕 | 81.0971° × 65.3704° | 78.1240° × 62.6969° | 5.4180% / 5.2877% |
| 左腕 | 81.8287° × 66.0443° | 78.2444° × 62.8459° | 6.5588% / 6.3888% |
| 头部 | 95.2601° × 63.2219° | 81.6902° × 65.9167° | 26.8040% / 26.8113% |

这些值是相同 K 的理想针孔视野，不是带畸变镜头的原始边界视野，也不能直接推出抓取或任务成功率。

## 外参与坐标系

当前 `T_parent_rgb_m` 与参考默认外参在仅将平移从 mm 换成 m 后，三路所有矩阵元素的最大绝对差均为 **0**。旋转无需求逆或另加安装角。

| 相机 | 父坐标系 | 平移（m） | 世界光学位姿 |
|---|---|---|---|
| 右腕 | `TCP_Link_R` 安装端 | (0.00211961167771889, 0.09119845517638979, 0.03725148394276182) | `T_world_flange_B @ T_flange_rgb_B` |
| 左腕 | `TCP_Link_L` 安装端 | (-0.0007043940851217335, -0.08750098578110607, 0.044302764749817754) | `T_world_flange_A @ T_flange_rgb_A` |
| 头部 | world / `base_link` | (0.1273603218756391, -0.0034876335486242415, 1.199981040227405) | 已发布 `T_base_cam` |

参考 `camera-extrinsics/backend.py` 在采集时先用模型 `T_world_base @ forward_kinematics` 形成世界安装端位姿；固定头部求解输出已经在 `base_link`。因此头部不能再次乘手臂安装矩阵。腕部 MuJoCo 父 body 已包含安装和 FK，亦不能再次乘安装矩阵。应用任务 TCP 的 `gripper_front` +185 mm 位于安装端下游，不属于相机外参。

额外离线核查：直接解析本项目 `assets/marvin_urdf/marvin_with_grippers.urdf`，按原始样本关节角逐关节计算 FK，与原始 `base_t_flange` 比较。头部 18 份 B 臂样本最大误差 0.003732 mm / 0.001146°；左腕 45 份为 0.004677 mm / 0.000770°；右腕 60 份为 0.005368 mm / 0.001300°。此结果验证坐标适配，不代表实机绝对标定精度。

光学坐标为 X 右、Y 下、Z 前；MuJoCo 相机为 X 右、Y 上、-Z 前。`cameras.py` 安装时右乘 `diag(1,-1,-1)`，读回位姿时再右乘同矩阵，恢复原始光学坐标。回归覆盖带 90° 世界安装旋转的双腕父 body 及固定头部。

## 投影与畸变边界

`model_builder.py:build()` 在编译前调用 `install_cameras()`。相机设置完整 `resolution`、`sensor_size`、`focal_pixel=[fx,fy]`、`principal_pixel=[W/2-cx-0.5,H/2-cy-0.5]`。`RgbdRenderer.capture()` 按每路尺寸渲染，并将相同 K 和光学外参提供给深度反投影。头部现为原生 640×480，因此正常 mapper 无需原先的上下各 60 行补边；通用 letterbox 支持继续保留。

锁定版本为 MuJoCo 3.10.0。其[官方相机文档](https://mujoco.readthedocs.io/en/3.10.0/XMLreference.html#body-camera)说明指定 sensor size 后使用内参、忽略 fovy；[官方实现](https://github.com/google-deepmind/mujoco/blob/3.10.0/src/engine/engine_vis_visualize.c)的 `mjv_cameraFrustum` 使用 `cam_intrinsic`，以及[官方主点测试](https://github.com/google-deepmind/mujoco/blob/3.10.0/test/engine/engine_vis_visualize_test.cc)用于核对主点偏移语义。本项目独立回归再验证整数像素中心与实际渲染斜面深度。

参考真实流水线的预览、录制和 VLA RGB **保留原生畸变**。外参检测 `engine.py:_rectify` 单独矫正图像，并使用不变的 K 求解；D405 的针孔 RGBD snapshot 也按公开 librealsense 去投影接口构造校正映射并保留 K。不能将原生图像直接当作零畸变图像。

本仿真本来输出理想针孔 RGBD，本次继续该合同：实测 K、实测分辨率、零输出畸变。没有把真实原生 RGB 当成矫正后图像，也没有实现镜头失真、传感器噪声、深度孔洞或逐像素原生预览匹配。`intrinsic_provenance.native_calibration` 保存原始 `rectified=false`、镜头模型和所有畸变系数；active `projection_model=ideal_pinhole` 明确输出含义。

三路来源 SHA256 分别为：

- 右腕：`e4a33a2fcb96b1edcd9aea40957e991dfc902a29a6173ee90908b87c19267dee`
- 左腕：`ca9c562b15e4fd96ffaafaf31fdd40ad6b0c5c7f7a7843f9cb136c725cfa9cdc`
- 头部：`5df5565d1093ff6310ab599531b0f90621686d5e6addbacc99850b4c8ef54d91`

`calibration_id=measured-rgb-pinhole-a85e3d8b0d1f2cc1` 同时由当前内参、外参、来源哈希和投影合同确定。旧芯明来源保存在明确的历史字段中，不再作为当前设备身份。修改前文件备份于 `data/observation-label-camera-baseline/resources/cameras/calibration.json`。

## 验证与生效

使用本地已有 `data/refactor-venv/Scripts/python.exe -X utf8`，MuJoCo 3.10.0：

1. 修改资源前运行 `-m pytest tests/test_camera_reference_calibration.py -q --tb=short`：**6 failed, 3 passed**。失败分别证明三路活跃 K 与实测来源不同，以及 native cam_intrinsic/头部分辨率错误；外参链三项原本通过。
2. 修改后运行 `-m pytest tests/test_camera_reference_calibration.py tests/test_cameras.py tests/test_camera_pixel_center.py -q --tb=short`：**17 passed**。覆盖来源 SN / 全部样本一致性、真实编译模型 cam_intrinsic / resolution、三维点的 native frustum 投影、三路外参链、实际 RGB/深度渲染及整数像素中心。
3. `-m pytest tests/test_camera_api.py -q --tb=short`：**2 passed**。实际应用模型、三路图像与场景代次失效通过；一个已有 Starlette/httpx 弃用警告，不属于相机错误。
4. `-m pytest tests/test_sensor_simulation.py tests/test_simulation.py -q --tb=short`：**16 passed**。实际仿真模型、传感器和场景生命周期回归通过。以上三个绿色命令合计 **35 passed**。

本阶段没有部署或重启远端服务，没有调用 Responses/Decisions 或控制实机。完整任务成功率和新视野下的视觉识别效果尚未验收。

相机参数在模型编译时加载，Renderer 另持有标定快照。单纯替换 JSON 不会更新已编译模型；`Simulation.reset()` 使用已确认 baseline 的旧 model，也不能更新内参。需在任务停止且状态可处理时重新配置场景使模型重编译，并重新确认 baseline，或重启项目进程后重新建立场景。旧观测和地图不得跨标定版本复用；不要在任务运行期间热替换标定文件。
