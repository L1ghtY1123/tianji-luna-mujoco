"""Typed, bounded robot skill programs produced once by Responses."""
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SkillName = Literal[
    "move_to_observe", "observe_scene", "identify_target",
    "identify_destination", "choose_arm", "build_scene", "generate_grasps",
    "plan_grasp", "execute_grasp", "verify_grasp", "plan_place", "execute_place", "verify_place",
    "prepare_unpack", "mock_unpack", "verify_unpack", "plan_transport", "execute_transport", "drop", "verify_drop",
]

PROGRAM_ARGUMENTS = {
    "identify_target": {"target_id": ["$target"]},
    "identify_destination": {"destination_id": ["$destination", "$table_zone", "$right_bin"]},
    "choose_arm": {"arm": ["auto", "A", "B"]},
}


class ProgramArgument(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: Literal["target_id", "destination_id", "arm"]
    value: Literal["$target", "$destination", "$table_zone", "$right_bin", "auto", "A", "B"]


class SkillCall(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["skill"]
    skill: SkillName
    arguments: list[ProgramArgument]


class Stop(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["stop"]
    reason: str


class IfResult(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["if_result"]
    condition: Literal["success", "unknown", "failure"]
    then_steps: list["ProgramNode"]
    else_steps: list["ProgramNode"]


class ForEachTarget(BaseModel):
    model_config = ConfigDict(extra="forbid")
    type: Literal["for_each_target"]
    body: list["ProgramNode"]
    between_targets: list["ProgramNode"]


ProgramNode = Annotated[SkillCall | Stop | IfResult | ForEachTarget, Field(discriminator="type")]
IfResult.model_rebuild()
ForEachTarget.model_rebuild()


class DemoProgram(BaseModel):
    model_config = ConfigDict(extra="forbid")
    version: Literal[1]
    steps: list[ProgramNode] = Field(min_length=1, max_length=64)

    @model_validator(mode="after")
    def bounded_program(self):
        count = 0

        def visit(nodes, depth=0, in_loop=False):
            nonlocal count
            if depth > 4:
                raise ValueError("program_nesting_exceeded")
            for node in nodes:
                count += 1
                if count > 64:
                    raise ValueError("program_node_budget_exceeded")
                if isinstance(node, ForEachTarget):
                    if in_loop:
                        raise ValueError("nested_target_loop")
                    if not node.body:
                        raise ValueError("empty_target_loop")
                    visit(node.body, depth + 1, True)
                    visit(node.between_targets, depth + 1, True)
                elif isinstance(node, IfResult):
                    visit(node.then_steps, depth + 1, in_loop)
                    visit(node.else_steps, depth + 1, in_loop)
                elif isinstance(node, SkillCall):
                    allowed = PROGRAM_ARGUMENTS.get(node.skill, {})
                    names = [arg.name for arg in node.arguments]
                    if len(set(names)) != len(names) or not set(names) <= set(allowed):
                        raise ValueError("invalid_skill_arguments:" + node.skill)
                    for arg in node.arguments:
                        if arg.value not in allowed[arg.name]:
                            raise ValueError("invalid_argument_binding:" + arg.name)
        visit(self.steps)
        return self


def order_unpack_observation(program):
    """Order an explicit sensing call before its read-only consumers; never invent calls."""
    result = program.model_copy(deep=True)
    changes = []

    def visit(nodes):
        for index, node in enumerate(nodes):
            if isinstance(node, ForEachTarget):
                visit(node.body)
                visit(node.between_targets)
            elif isinstance(node, IfResult):
                visit(node.then_steps)
                visit(node.else_steps)
            elif isinstance(node, SkillCall) and node.skill == 'verify_unpack':
                following = index + 1
                # Preserve the model's success/failure guard before any continuation.
                if following < len(nodes) and isinstance(nodes[following], IfResult):
                    guard = nodes[following]
                    if guard.condition != 'success' or guard.then_steps:
                        continue
                    following += 1
                position = following
                while position < len(nodes) and isinstance(nodes[position], SkillCall):
                    name = nodes[position].skill
                    if name == 'observe_scene':
                        break
                    if name not in {'identify_target', 'identify_destination', 'choose_arm'}:
                        raise ValueError('program_unpack_observation_required')
                    position += 1
                if (position >= len(nodes) or not isinstance(nodes[position], SkillCall)
                        or nodes[position].skill != 'observe_scene'):
                    raise ValueError('program_unpack_observation_required')
                if position != following:
                    nodes.insert(following, nodes.pop(position))
                    changes.append({'skill': 'observe_scene', 'from_index': position, 'to_index': following,
                                    'reason': 'black_target_selection_requires_current_measured_geometry'})

    visit(result.steps)
    return result, changes
