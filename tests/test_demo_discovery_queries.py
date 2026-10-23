"""Demo query vocabulary is scoped to validated task IDs and recognized types."""
import json
from copy import deepcopy
from pathlib import Path

import pytest
from test_demo_perception import demo_discovery, discovery_output
from test_perception import capture_fixture

from luna_mujoco.visual_api import VisualInventoryError
from luna_mujoco.visual_discovery import DemoDiscoveredEntity, DiscoveredEntity


def test_frozen_chinese_inventory_normalizes_colors_and_preserves_raw_answer():
    body = json.loads((Path(__file__).parent / "fixtures/demo-discovery-8bfd.json").read_text(encoding="utf-8"))
    capture, _ = capture_fixture()
    requests, events = [], []
    discovery = demo_discovery(body, requests)
    discovery.audit = events.append

    result = discovery.discover(capture, instruction="move all pink objects")

    task, entities = result["demo_task"], result["entities"]
    assert entities[task["source_bin_id"]]["color"] == "blue"
    assert entities[task["destination_table_id"]]["color"] == "gray"
    assert len(task["source_object_ids"]) == len(task["transport_ids"]) == 3
    assert all(entities[identifier]["color"] == "pink" for identifier in task["source_object_ids"])
    assert all(item["segmentation_query"]["color"] == item["color"]
               for item in entities.values() if "segmentation_query" in item)
    assert result["demo_program"] == body["program"]
    raw_text = events[0]["response"]["output"][0]["content"][0]["text"]
    assert json.loads(raw_text) == body
    assert body["entities"][1]["color"] == "蓝色"
    assert len(requests) == 1


@pytest.mark.parametrize("color,expected", [
    (" BLUE ", "blue"), ("蓝", "blue"), ("蓝色", "blue"), ("藍", "blue"), ("藍色", "blue"),
    (" Pink\t", "pink"), ("粉色", "pink"), ("粉红色", "pink"), ("粉紅色", "pink"),
    (" GRAY ", "gray"), ("grey", "gray"), ("灰", "gray"), ("灰色", "gray"),
    (" Red ", "red"), ("红色", "红色"), ("unknown", "unknown"), ("未知", "未知"),
    ("蓝色或红色", "蓝色或红色"), ("淡粉色", "淡粉色"), ("pinkish", "pinkish"),
])
def test_color_aliases_are_exact_and_demo_only(color, expected):
    entity = {**discovery_output()["entities"][2], "color": color}
    assert DemoDiscoveredEntity.model_validate(entity).color == expected
    entity.pop("reference_camera_id")
    assert DiscoveredEntity.model_validate(entity).color == color


@pytest.mark.parametrize("color", ["unknown", "未知", "red", "红色", "blue and red", "蓝色或红色"])
def test_source_color_is_not_inferred_from_kind_reference_or_prompt(color):
    capture, _ = capture_fixture()
    body = discovery_output()
    body["entities"][2]["color"] = color
    with pytest.raises(VisualInventoryError, match="demo_blue_source_bin_unconfirmed"):
        demo_discovery(body, []).discover(capture, instruction="move all pink objects from the blue bin")


def test_reference_view_guidance_prefers_visible_wrists_without_inventing_entities():
    capture, _ = capture_fixture()
    requests = []
    demo_discovery(discovery_output(), requests).discover(capture, instruction="move all pink objects")
    content = requests[0]["input"][0]["content"]
    instructions = json.loads(content[0]["text"])["instructions"]
    assert "Prefer camera 2 for the source box and its pink contents" in instructions
    assert "prefer camera 1 for the separate right box" in instructions
    assert "when that entity is visibly identifiable in the respective wrist image" in instructions
    assert "Do not require camera 3 to see all source objects" in instructions
    assert "Camera roles do not prove an entity is present or establish identity" in instructions
    assert "Other entities can use whichever camera shows them" in instructions
    assert sum(part["type"] == "input_image" for part in content) == 3


@pytest.mark.parametrize("blue,pink", [
    ("blue", "pink"), ("BLUE", "Pink"), (" Blue ", " PINK\t"), ("蓝色", "粉色"), ("藍色", "粉紅色"),
])
def test_canonical_queries_preserve_inventory_and_include_unselected_sources(blue, pink):
    capture, _ = capture_fixture()
    body = discovery_output(1)
    body["entities"][2].update(color=blue, segmentation_prompt="open blue plastic bin")
    body["entities"][5].update(color=pink, segmentation_prompt="pink cylinder inside the blue bin")
    original = deepcopy(body)
    requests = []
    discovery = demo_discovery(body, requests)

    result = discovery.discover(capture, instruction="move one pink object")
    entities, task = result["entities"], result["demo_task"]
    source = entities[task["source_bin_id"]]
    cylinder_id = next(identifier for identifier in task["source_object_ids"]
                       if entities[identifier]["kind"] == "cylinder")
    cylinder = entities[cylinder_id]

    assert source["segmentation_query"] == {"kind": "bin", "color": "blue", "prompt": "blue box"}
    assert cylinder["segmentation_query"] == {"kind": "cylinder", "color": "pink", "prompt": "pink object"}
    assert cylinder_id not in task["transport_ids"]
    assert cylinder["visual_membership"] == "excluded"
    assert cylinder["initial_parent_id"] == task["source_bin_id"]
    assert cylinder["bbox"] == original["entities"][5]["bbox"]
    assert cylinder["reference_camera_id"] == "2"
    assert cylinder["description"] == original["entities"][5]["description"]
    assert result["reference"]["entities"] == entities
    assert discovery.discover(capture, instruction="move one pink object")["entities"] == entities
    assert len(requests) == 1
    assert body == original


@pytest.mark.parametrize("kind,prompt", [("cube", "pink cube"), ("sphere", "pink ball"), ("object", "pink toy")])
def test_source_types_keep_kind_and_share_shape_neutral_queries(kind, prompt):
    capture, _ = capture_fixture()
    body = discovery_output()
    body["entities"][5].update(kind=kind, segmentation_prompt=prompt)
    result = demo_discovery(body, []).discover(capture, instruction="move all pink objects")
    item = next(entity for entity in result["entities"].values() if entity["description"] == "bottom")
    assert item["kind"] == kind
    assert item["segmentation_query"]["prompt"] == "pink object"
    assert len(item["segmentation_prompts"]) == 4


def test_matching_colors_and_types_outside_task_source_keep_their_queries():
    capture, _ = capture_fixture()
    body = discovery_output()
    for original, ref, prompt in [(body["entities"][2], "other-bin", "blue storage bin"),
                                  (body["entities"][5], "other-cylinder", "pink cylinder")]:
        body["entities"].append({**deepcopy(original), "ref": ref, "description": ref,
                                 "parent_ref": "large", "segmentation_prompt": prompt})
    result = demo_discovery(body, []).discover(capture, instruction="move all pink source objects")
    by_name = {entity["description"]: entity for entity in result["entities"].values()}
    assert by_name["other-bin"]["segmentation_query"]["prompt"] == "blue storage bin"
    assert by_name["other-cylinder"]["segmentation_query"]["prompt"] == "pink cylinder"
    assert by_name["other-cylinder"]["id"] not in result["demo_task"]["source_object_ids"]


@pytest.mark.parametrize("field,value,reason", [
    ("kind", "teapot", "visual_discovery_invalid_response"),
    ("segmentation_prompt", "", "visual_discovery_invalid_response"),
])
def test_canonicalization_does_not_rescue_invalid_source_output(field, value, reason):
    capture, _ = capture_fixture()
    body = discovery_output()
    body["entities"][5][field] = value
    with pytest.raises(VisualInventoryError, match=reason):
        demo_discovery(body, []).discover(capture, instruction="move all pink objects")


def test_initial_inventory_and_requested_count_have_no_legacy_four_or_sixty_four_limit():
    body = discovery_output()
    sources = [entity for entity in body['entities'] if entity['color'] == 'pink']
    body['entities'] = [entity for entity in body['entities'] if entity['color'] != 'pink']
    for index in range(65):
        body['entities'].append({**deepcopy(sources[0]), 'ref': 'pink-' + str(index),
                                 'bbox': [.1, .1, .2, .2]})
    body['task'].update(scope='count', requested_count=65)
    capture, _ = capture_fixture()
    result = demo_discovery(body, []).discover(capture, instruction='move 65 pink objects')
    assert len(result['demo_task']['transport_ids']) == len(result['demo_task']['source_object_ids']) == 65
    assert result['demo_task']['requested_count'] == 65


def test_count_subset_keeps_initial_display_numbers_for_all_original_sources():
    capture, _ = capture_fixture()
    result = demo_discovery(discovery_output(1), []).discover(capture, instruction='move one')
    ids = result['demo_task']['source_object_ids']
    assert [result['entities'][key]['display_id'] for key in ids] == [1, 2, 3]
    assert [result['reference']['entities'][key]['display_id'] for key in ids] == [1, 2, 3]
    assert [result['entities'][key]['visual_membership'] for key in ids] == ['required', 'excluded', 'excluded']
