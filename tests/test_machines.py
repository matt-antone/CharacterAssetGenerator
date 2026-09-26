import json

import pytest

from cag import comfy, machines
from cag.machines import MachineError, load_machine, machine_names, materialise, patch_workflow


def placeholders(workflow: dict) -> set[str]:
    return {value for node in workflow.values() for value in node.get("inputs", {}).values()
            if isinstance(value, str) and value.startswith("$")}


def links(workflow: dict) -> list[str]:
    return [str(value[0]) for node in workflow.values() for value in node.get("inputs", {}).values()
            if isinstance(value, list) and len(value) == 2]


def test_the_shipped_profiles_are_cloud_local16_and_smoke4():
    assert machine_names() == ["cloud", "local16", "smoke4"]
    cloud = load_machine("cloud")
    assert cloud.verified and cloud.backend == "comfy"
    assert (cloud.width, cloud.height, cloud.rate, cloud.max_length, cloud.steps) == (576, 864, 16, 81, 6)
    assert (cloud.restyle_resolution, cloud.restyle_steps, cloud.seed) == (1248, 40, 1234)
    assert not any(cloud.patch(stage) for stage in machines.STAGES), "the graphs carry cloud's files"
    assert [load_machine(name).verified for name in ("local16", "smoke4")] == [False, False]


def test_an_unknown_machine_lists_the_ones_there_are():
    with pytest.raises(MachineError, match="cloud, local16, smoke4"):
        load_machine("rtx5090")


def test_the_shipped_placeholders_are_exactly_the_documented_set():
    for stage in machines.STAGES:
        shipped = comfy.load_workflow(machines.workflow_path(stage))
        assert placeholders(shipped) == machines.PLACEHOLDERS[stage], stage


@pytest.mark.parametrize("name", ["cloud", "local16", "smoke4"])
def test_every_profile_patches_every_graph_and_leaves_nothing_unfilled(name, tmp_path):
    machine = load_machine(name)
    for stage in machines.STAGES:
        path = materialise(machine, stage, tmp_path)
        assert path == tmp_path / name / f"{stage}.json"
        graph = json.loads(path.read_text())
        assert set(links(graph)) <= set(graph), "every link lands on a node"

        slots = comfy.reference_slots(graph)
        extra = {**machine.placeholders(stage), **({"$length": 41} if stage == "video" else {})}
        filled = comfy.fill(graph, "a dancer", [f"ref{n}.png" for n in range(slots)], 7,
                            comfy.PORTRAIT, extra=extra)
        assert not placeholders(filled), f"{name} {stage} left {placeholders(filled)}"
        if stage == "video":
            assert filled["S"]["inputs"]["width"] == machine.width, "the profile beats the prompt"
            assert filled["K"]["inputs"]["seed"] == machine.seed


def test_each_stage_takes_its_own_numbers_from_the_profile():
    smoke = load_machine("smoke4")
    assert smoke.placeholders("video") == {"$width": 256, "$height": 384, "$steps": 2, "$seed": 1234}
    assert "$seed" not in smoke.placeholders("restyle"), "draw moves the restyle's seed on"
    assert smoke.placeholders("mask") == {}


def test_smoke4_loads_scail_as_a_gguf_with_only_its_file_name(tmp_path):
    graph = json.loads(materialise(load_machine("smoke4"), "video", tmp_path).read_text())
    assert graph["1"] == {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "SCAIL-2-Q2_K.gguf"}}
    assert graph["2"]["inputs"]["model"] == ["1", 0], "the LoRA chain still starts at the loader"
    assert graph["5"]["inputs"] == {
        "clip_name": "umt5_xxl_fp8_e4m3fn_scaled.safetensors", "type": "wan", "device": "cpu"
    }
    restyle = json.loads(materialise(load_machine("smoke4"), "restyle", tmp_path).read_text())
    assert restyle["clip"]["inputs"]["device"] == "cpu"
    assert restyle["unet"]["inputs"]["weight_dtype"] == "fp8_e4m3fn"


def test_local16_loads_the_q5_gguf():
    shipped = comfy.load_workflow(machines.workflow_path("video"))
    graph = patch_workflow(shipped, load_machine("local16").patch("video"))
    assert graph["1"]["inputs"] == {"unet_name": "SCAIL-2-Q5_K_M.gguf"}
    assert shipped["1"]["class_type"] == "UNETLoader", "the shipped graph is left alone"


def test_a_patch_aimed_at_a_node_the_graph_lacks_is_refused():
    shipped = comfy.load_workflow(machines.workflow_path("video"))
    with pytest.raises(MachineError, match="node 'unet'"):
        patch_workflow(shipped, {"unet": {"inputs": {"unet_name": "x.gguf"}}})


def test_a_misspelt_input_is_refused_rather_than_dropped():
    shipped = comfy.load_workflow(machines.workflow_path("video"))
    with pytest.raises(MachineError, match="no input devcie"):
        patch_workflow(shipped, {"5": {"inputs": {"devcie": "cpu"}}})


def test_a_replacement_that_orphans_a_link_is_refused():
    graph = {"a": {"class_type": "X", "inputs": {}}, "b": {"class_type": "Y", "inputs": {"m": ["a", 0]}}}
    patched = patch_workflow(graph, {"a": {"class_type": "Z", "inputs": {"n": 1}}})
    assert patched["a"] == {"class_type": "Z", "inputs": {"n": 1}}
    with pytest.raises(MachineError, match="missing node 'gone'"):
        patch_workflow(graph, {"b": {"inputs": {"m": ["gone", 0]}}})


def test_a_stage_graph_can_be_named_by_environment(tmp_path, monkeypatch):
    own = tmp_path / "mine.json"
    own.write_text(json.dumps({"1": {"class_type": "T", "inputs": {"text": "$prompt"}}}))
    monkeypatch.setenv("CAG_COMFY_MASK_WORKFLOW", str(own))
    assert machines.workflow_path("mask") == own
    written = materialise(load_machine("cloud"), "mask", tmp_path / "graphs")
    assert json.loads(written.read_text()) == json.loads(own.read_text())
    with pytest.raises(MachineError, match="no stage"):
        machines.workflow_path("scail")


def test_a_seed_variable_overrides_every_profile(monkeypatch):
    monkeypatch.setenv("CAG_VIDEO_SEED", "7")
    assert load_machine("cloud").seed == 7


def test_a_profile_the_graphs_cannot_take_is_refused(tmp_path):
    profile = json.loads((machines.MACHINES / "smoke4.json").read_text())
    profile["video"].update(width=250, max_length=10)
    (tmp_path / "odd.json").write_text(json.dumps(profile))
    with pytest.raises(MachineError, match=r"250x384 is not a multiple of 32; max_length 10 is not 4k\+1"):
        load_machine("odd", root=tmp_path)
    del profile["restyle"]
    (tmp_path / "short.json").write_text(json.dumps(profile))
    with pytest.raises(MachineError, match="no 'restyle'"):
        load_machine("short", root=tmp_path)
