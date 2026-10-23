"""plan JSON 校验与实例展开（文档 §2.1.1 schema）。"""
from algorithm.planner.schema import expand_instances, parse_plan


def base_raw(**over):
    raw = {
        "scene_summary": "demo",
        "background_repair": {"mode": "scene", "reason": "sky visible"},
        "text_corrections": [],
        "raster_text_ids": ["text_001"],
        "queries": [{
            "id": "main_panel", "name": "Main", "kind": "panel",
            "role": "container", "element_repair_mode": "image",
            "parent_query_id": None, "z_order": 1,
            "geometry_hints": [{"bbox_norm": [0.1, 0.1, 0.5, 0.5],
                                "positive_points_norm": [[0.2, 0.2]],
                                "negative_points_norm": []}],
        }],
    }
    raw.update(over)
    return raw


def test_valid_plan_parses():
    plan, warns = parse_plan(base_raw(), ["text_001", "text_002"])
    assert not warns
    assert plan.background_mode == "scene"
    assert plan.raster_text_ids == ["text_001"]
    assert len(plan.queries) == 1


def test_invalid_enum_drops_query():
    raw = base_raw()
    raw["queries"][0]["kind"] = "spaceship"
    plan, warns = parse_plan(raw, ["text_001"])
    assert plan.queries == [] and warns


def test_unknown_raster_id_dropped():
    plan, warns = parse_plan(base_raw(), ["text_001"])
    plan2, warns2 = parse_plan(base_raw(raster_text_ids=["text_009"]), ["text_001"])
    assert plan2.raster_text_ids == [] and warns2


def test_bad_hints_drop_query():
    raw = base_raw()
    raw["queries"][0]["geometry_hints"] = [
        {"bbox_norm": [0.5, 0.5, 0.1, 0.1], "positive_points_norm": [[0.2, 0.2]]},  # 逆序收敛后有效
        {"bbox_norm": [0.1, 0.1, 0.2, 0.2], "positive_points_norm": []},  # 无正点 -> 丢弃
    ]
    plan, warns = parse_plan(raw, ["text_001"])
    assert len(plan.queries[0].hints) == 1


def test_parent_cycle_broken():
    raw = base_raw()
    raw["queries"].append({
        "id": "child", "name": "c", "kind": "icon", "role": "foreground",
        "element_repair_mode": "none", "parent_query_id": "grandchild",
        "z_order": 3,
        "geometry_hints": [{"bbox_norm": [0.2, 0.2, 0.3, 0.3],
                            "positive_points_norm": [[0.25, 0.25]]}]})
    raw["queries"].append({
        "id": "grandchild", "name": "g", "kind": "badge", "role": "foreground",
        "element_repair_mode": "none", "parent_query_id": "child",
        "z_order": 4,
        "geometry_hints": [{"bbox_norm": [0.2, 0.2, 0.3, 0.3],
                            "positive_points_norm": [[0.25, 0.25]]}]})
    plan, warns = parse_plan(raw, ["text_001"])
    by_id = {q.id: q for q in plan.queries}
    assert any("成环" in w for w in warns)
    assert by_id["child"].parent_query_id is None or \
        by_id["grandchild"].parent_query_id is None


def test_expand_instances_naming():
    raw = base_raw()
    raw["queries"][0]["geometry_hints"].append(
        {"bbox_norm": [0.5, 0.5, 0.9, 0.9], "positive_points_norm": [[0.6, 0.6]]})
    plan, _ = parse_plan(raw, ["text_001"])
    insts = expand_instances(plan)
    assert [i.instance_id for i in insts] == ["main_panel#1", "main_panel#2"]
