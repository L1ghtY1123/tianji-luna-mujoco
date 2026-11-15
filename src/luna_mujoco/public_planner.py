"""Composition only: official planners own all internal algorithm calls."""
import time
from contextlib import contextmanager

from .planning_timing import measured


class PublicPlanner:
    def __init__(self, sdk, settings, *, batch=False, grasp_contact_links=()):
        self.sdk = sdk
        self.planning_settings = settings
        self.batch = batch
        self.last_grasp_result = None
        self.last_pose_result = None
        self.last_grasp_wall_s = None
        self._planning_scene = None
        self.scene_restore_failed = False
        self.grasp_contact_links = tuple(grasp_contact_links)
        self.collision_restore_failed = False

    def set_planning_scene(self, original, reduced, original_disabled, reduced_disabled):
        """Called after the full collision world has been successfully uploaded."""
        self._planning_scene = None if reduced is None else (
            original, reduced, original_disabled, reduced_disabled)
        self.scene_restore_failed = False

    def _load_scene(self, scene, disabled):
        self.sdk.update_world(scene)
        data = self.sdk.scene_collision_checker.data
        for kind, geometries in (('mesh', data.meshes), ('cuboid', data.cuboids)):
            for name in disabled.get(kind, ()):
                geometries.set_enabled(name, enabled=False, env_idx=0)

    @contextmanager
    def optimizer_scene(self):
        if self.collision_restore_failed:
            raise RuntimeError('grasp_collision_links_not_restored')
        if self.scene_restore_failed:
            raise RuntimeError('original_collision_scene_not_restored')
        if self._planning_scene is None:
            yield
            return
        original, reduced, original_disabled, reduced_disabled = self._planning_scene
        try:
            self._load_scene(reduced, reduced_disabled)
            yield
        finally:
            self.scene_restore_failed = True
            self._load_scene(original, original_disabled)
            self.scene_restore_failed = False

    def __getattr__(self, name):
        # Compatibility with the existing transport wrapper. No methods or
        # attributes are installed on the SDK instance; internal self calls
        # always resolve to the original NVIDIA class.
        return getattr(self.sdk, name)

    @measured('public_plan_pose')
    def plan_pose(self, goal_tool_poses, current_state, *args, **kwargs):
        with self.optimizer_scene():
            self.last_pose_result = self.sdk.plan_pose(goal_tool_poses, current_state)
        return self.last_pose_result

    @measured('public_plan_grasp')
    def plan_grasp(self, *args, **kwargs):
        self.last_grasp_result = None
        started = time.perf_counter()
        # The pinned public SDK restores these links for approach, and masks
        # them for goalset, grasp and lift. Exact command checks retain the
        # environment and non-contact links before any trajectory is executed.
        # Two-phase placement has its own collision policy and keeps its input.
        contacts = list(self.grasp_contact_links) if kwargs.get('plan_grasp_to_lift', True) else []
        if contacts:
            kwargs['disable_collision_links'] = contacts
        with self.optimizer_scene():
            try:
                self.last_grasp_result = self.sdk.plan_grasp(*args, **kwargs)
            finally:
                if contacts:
                    self.collision_restore_failed = True
                    self.sdk.enable_link_collision(contacts)
                    self.collision_restore_failed = False
        self.last_grasp_wall_s = time.perf_counter() - started
        return self.last_grasp_result
