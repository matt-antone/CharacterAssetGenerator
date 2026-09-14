from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from cag import animation, mask
from cag.motion import load_motion
from cag.spec import load_spec
from tests.test_static_sheet import flat_cutout

SAMPLE = "/Users/matthewantone/Development/MotionArtist/work/sample/motion.json"
NOTE = "The microphone stays in the character-right hand for every frame."


def fake_draw(prompt, out_path, references=(), **kwargs):
    """Each render differs, so GIF assembly cannot collapse identical frames."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shade = 20 + 10 * (len(fake_draw.calls) % 12)
    image = Image.new("RGBA", (100, 200), (255, 0, 255, 255))
    image.paste((shade, shade, shade, 255), (40, 20, 60, 180))
    image.save(out_path)
    fake_draw.calls.append({"prompt": prompt, "out": out_path, "refs": list(references)})
    return out_path


@pytest.fixture
def run(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    graph = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw
    )
    return graph.invoke(
        {
            "spec": load_spec("specs/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": load_motion(SAMPLE),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )


def order(calls):
    return [int(call["out"].stem) for call in calls]


def test_keys_and_pilots_are_drawn_before_any_inbetween(run):
    drawn = order(fake_draw.calls)
    locked = [0, 2, 7, 9, 11, 13]
    assert drawn[: len(locked)] == locked
    assert sorted(drawn) == list(range(16))


def sources(call):
    """Neighbour frame indices referenced, ignoring key art, detail and pose."""
    return [
        int(Path(p).stem)
        for p in call["refs"]
        if Path(p).parent.name == "dance" and Path(p).parent.parent.name == "source"
    ]


def pose_ref(call):
    return Path(call["refs"][-1])


def test_keyframes_reference_the_key_art_and_their_own_pose(run):
    key_art = fake_draw.calls[0]["refs"][0]
    for call in fake_draw.calls[:6]:
        assert call["refs"][0] == key_art
        assert sources(call) == []
        assert pose_ref(call).parts[-3] == "poses"
        assert pose_ref(call).stem == call["out"].stem


def test_inbetweens_reference_the_key_art_and_both_neighbours(run):
    by_index = {int(c["out"].stem): c for c in fake_draw.calls}
    # Frame 1 sits between locked frames 0 and 2.
    assert sources(by_index[1]) == [0, 2]
    # Frame 8 follows locked frame 7 and runs up to pilot frame 9.
    assert sources(by_index[8]) == [7, 9]


def test_the_last_inbetween_closes_onto_frame_zero(run):
    by_index = {int(c["out"].stem): c for c in fake_draw.calls}
    assert sources(by_index[15]) == [14, 0]


def test_the_pose_skeleton_is_always_the_last_reference(run):
    for call in fake_draw.calls:
        assert pose_ref(call).parts[-3] == "poses"
        assert pose_ref(call).stem == call["out"].stem
        assert "stick-figure skeleton" in call["prompt"]


def test_every_frame_carries_the_detail_sample_before_the_pose(run):
    for call in fake_draw.calls:
        names = [Path(p).name for p in call["refs"]]
        assert "detail-level-04.png" in names
        assert names.index("detail-level-04.png") == len(names) - 2
        assert "detail level 4" in call["prompt"]
        assert "Capcom Street Fighter 2" in call["prompt"]


def test_a_sheet_without_poses_draws_without_one(tmp_path, monkeypatch):
    """An older motion.json has no landmarks; the set still renders."""
    import json

    from cag import animation as animation_module
    from cag.motion import load_motion

    raw = json.loads(Path(SAMPLE).read_text())
    for frame in raw["frames"]:
        frame.pop("pts", None)
    stripped = tmp_path / "motion.json"
    stripped.write_text(json.dumps(raw))

    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    animation_module.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw
    ).invoke(
        {
            "spec": load_spec("specs/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": load_motion(stripped),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    assert len(fake_draw.calls) == 16
    assert all("stick-figure skeleton" not in c["prompt"] for c in fake_draw.calls)


def test_every_frame_prompt_carries_the_directors_note_and_view(run):
    for call in fake_draw.calls:
        assert NOTE in call["prompt"]
        assert "facing the viewer square-on" in call["prompt"]


def test_produces_sixteen_registered_cells(run):
    assert sorted(run["cells"]) == list(range(16))
    with Image.open(run["cells"][0]) as cell:
        assert cell.size == (480, 560)
