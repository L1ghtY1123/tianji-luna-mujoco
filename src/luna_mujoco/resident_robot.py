"""Fixed-size collision models updated through public cuRobo geometry APIs."""
import json
import time
from copy import deepcopy

import numpy as np

from .carrying import carrying_geometry, carrying_spheres, carrying_touch_links, configure_gripper_feedback

ATTACHMENT_CAPACITY = 31


def reserve_attachment_links(robot):
    result = deepcopy(robot)
    kin = result['robot_cfg']['kinematics']
    kin['collision_link_names'] = list(kin['collision_link_names'])
    for arm, side in (('A', 'L'), ('B', 'R')):
        link = 'jev_carried_object_' + arm
        kin.setdefault('extra_links', {})[link] = {
            'fixed_transform': [0, 0, 0, 1, 0, 0, 0], 'joint_name': 'jev_carry_joint_' + arm,
            'joint_type': 'FIXED', 'link_name': link, 'parent_link_name': 'TCP_Link_' + side}
        if link not in kin['collision_link_names']:
            kin['collision_link_names'].append(link)
        kin['collision_spheres'][link] = [{'center': [0., 0., 0.], 'radius': -100.}
                                          for _ in range(ATTACHMENT_CAPACITY)]
        if isinstance(kin.get('collision_sphere_buffer'), dict):
            kin['collision_sphere_buffer'][link] = 0.
        kin.setdefault('self_collision_buffer', {})[link] = 0.
        contacts = [n for n in carrying_touch_links(arm) if n in kin['collision_link_names']]
        ignored = kin.setdefault('self_collision_ignore', {})
        ignored[link] = contacts
        for name in contacts:
            if link not in ignored.setdefault(name, []):
                ignored[name].append(link)
    return result


class ResidentRobotGeometry:
    def __init__(self, robot, urdf_path, *, peer_robot=None, peer_links=()):
        self.robot = reserve_attachment_links(robot)
        self.urdf_path = urdf_path
        self.peer_robot = peer_robot
        self.peer_links = list(peer_links)
        # Builders instantiate the same open, empty reserved geometry before warmup.
        self.key = json.dumps([{'A': 0., 'B': 0.}, None, None], sort_keys=True)
        self.feedback_key = json.dumps({'A': 0., 'B': 0.}, sort_keys=True)

    def update(self, backend, feedback, binding):
        from curobo.kinematics import Kinematics, KinematicsCfg
        from curobo.types import Pose

        started = time.perf_counter()
        feedback = feedback or {'A': 0., 'B': 0.}
        spheres = None if binding is None else carrying_spheres(binding)
        key = json.dumps([feedback, None if binding is None else binding.arm, spheres], sort_keys=True)
        if key == self.key:
            backend.carrying_scene_token = None if binding is None else binding.scene_token
            self._evidence(backend, binding)
            return False
        planner = backend.motion_planner
        targets = [planner.kinematics, backend._motion_clearance_checker.kinematics]
        if not backend.ik_only:
            targets.extend([planner.ik_solver.kinematics, planner.graph_planner.kinematics])
        feedback_key = json.dumps(feedback, sort_keys=True)
        changed_feedback = feedback_key != self.feedback_key
        if changed_feedback:
            robot = configure_gripper_feedback(self.robot, feedback, self.urdf_path)
            cfg = KinematicsCfg.from_data_dict(robot, tool_frames=list(planner.tool_frames), device_cfg=planner.device_cfg)
        # Reparse locked-joint transforms only when measured fingers change.
        # Attachment alone uses the public fixed-capacity sphere update API.
        seen = set()
        for target in targets:
            params = target.config.kinematics_config
            if id(params) not in seen:
                if changed_feedback:
                    target.update_kinematics_config(cfg.kinematics_config)
                for arm in ('A', 'B'):
                    balls = backend._torch.zeros((ATTACHMENT_CAPACITY, 4), **planner.device_cfg.as_torch_dict())
                    balls[:, 3] = -100.
                    if binding is not None and binding.arm == arm:
                        if len(spheres) > ATTACHMENT_CAPACITY:
                            raise ValueError('attachment_exceeds_resident_capacity')
                        balls[:len(spheres)] = backend._torch.as_tensor(spheres, **planner.device_cfg.as_torch_dict())
                    params.update_link_spheres('jev_carried_object_' + arm, balls)
                seen.add(id(params))
        if changed_feedback and self.peer_robot is not None:
            peer = configure_gripper_feedback(self.peer_robot, feedback, self.urdf_path)
            peer_cfg = KinematicsCfg.from_data_dict(peer, device_cfg=planner.device_cfg)
            backend._peer_kinematics.update_kinematics_config(peer_cfg.kinematics_config)
            reference = Kinematics(KinematicsCfg.from_data_dict(robot,
                tool_frames=self.peer_links, device_cfg=planner.device_cfg))
            zero = backend._torch.zeros((1, len(reference.joint_names)), **planner.device_cfg.as_torch_dict())
            poses = reference.get_link_poses(zero, self.peer_links)
            matrices = Pose(position=poses.position.reshape(-1, 3),
                            quaternion=poses.quaternion.reshape(-1, 4)).get_matrix()
            backend._peer_baseline_inverse_by_link = {name: backend._torch.linalg.inv(
                matrices[i].to(dtype=backend._torch.float64)).detach().clone() for i, name in enumerate(self.peer_links)}
        if not backend.ik_only:
            planner.graph_planner.reset_buffer()
        backend.carrying_link = None if binding is None else 'jev_carried_object_' + binding.arm
        backend.carrying_spheres_tool = None if spheres is None else np.asarray(spheres)
        backend.carrying_scene_token = None if binding is None else binding.scene_token
        self.key = key
        self.feedback_key = feedback_key
        self._evidence(backend, binding)
        backend.robot_geometry_update_s = time.perf_counter() - started
        return True

    def detach(self, backend):
        # Do not parse eleven robots during a release. Clear fixed slots now;
        # the next request supplies and updates the actual finger feedback.
        planner = backend.motion_planner
        targets = [planner.kinematics, backend._motion_clearance_checker.kinematics]
        if not backend.ik_only:
            targets.extend([planner.ik_solver.kinematics, planner.graph_planner.kinematics])
        seen = set()
        for target in targets:
            params = target.config.kinematics_config
            if id(params) in seen:
                continue
            balls = backend._torch.zeros((ATTACHMENT_CAPACITY, 4), **planner.device_cfg.as_torch_dict())
            balls[:, 3] = -100.
            for arm in ('A', 'B'):
                params.update_link_spheres('jev_carried_object_' + arm, balls)
            seen.add(id(params))
        backend.carrying_link = backend.carrying_spheres_tool = backend.carrying_scene_token = None
        self.key = None
        self._evidence(backend, None)

    @staticmethod
    def _evidence(backend, binding):
        if not hasattr(backend, 'effective_config'):
            return
        geometry = None if binding is None else carrying_geometry(binding)
        if geometry is not None:
            geometry.pop('spheres', None)
        backend.effective_config.update(optimizer_attachment=binding is not None,
            carried_link=None if binding is None else 'jev_carried_object_' + binding.arm,
            carrying_owner_id=None if binding is None else binding.owner_id,
            attachment_geometry=geometry, resident_attachment_capacity=ATTACHMENT_CAPACITY,
            gripper_configuration_source='measured_feedback')
