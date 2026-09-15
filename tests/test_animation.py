from pathlib import Path

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from cag import animation, mask
from cag.geometry import ANIM_CONTACT_ROW, anim_subject_height_px
from cag.motion import load_motion
from cag.skeleton import pose_extent, stature
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
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw, sheet_mode=False
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


def test_inbetweens_continue_from_the_frame_before_them_only(run):
    """The frame ahead is left off so it cannot vote against the pose."""
    by_index = {int(c["out"].stem): c for c in fake_draw.calls}
    assert sources(by_index[1]) == [0]
    assert sources(by_index[8]) == [7]
    assert sources(by_index[15]) == [14]


def test_no_inbetween_carries_more_than_one_neighbour(run):
    for call in fake_draw.calls:
        assert len(sources(call)) <= 1


def test_the_pose_skeleton_is_always_the_last_reference(run):
    for call in fake_draw.calls:
        assert pose_ref(call).parts[-3] == "poses"
        assert pose_ref(call).stem == call["out"].stem
        assert "stick-figure skeleton" in call["prompt"]


def test_every_frame_names_its_detail_level_and_style(run):
    from cag.style import DEFAULT_DETAIL_LEVEL, detail_frame

    sample = detail_frame(DEFAULT_DETAIL_LEVEL)
    for call in fake_draw.calls:
        assert f"detail level {DEFAULT_DETAIL_LEVEL}" in call["prompt"]
        assert "32-bit arcade sprite art" in call["prompt"]
        names = [Path(p).name for p in call["refs"]]
        if sample:
            # The pose must stay last, so the sample sits just before it.
            assert names.index(sample.name) == len(names) - 2


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
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw, sheet_mode=False
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


def cell_height(path):
    with Image.open(path) as cell:
        left, top, right, bottom = mask.subject_box(cell)
    return bottom - top, bottom - 1


def test_every_cell_stands_on_the_animation_contact_row(run):
    for path in run["cells"].values():
        assert cell_height(path)[1] == ANIM_CONTACT_ROW


def test_each_cell_is_scaled_by_the_pose_of_its_own_frame(run):
    """Every frame is drawn at the same size here, and the poses differ, so the
    cells must differ too — by exactly what each frame's own skeleton asks for.
    Hand a frame the wrong pose and its height stops matching."""
    motion = load_motion(SAMPLE)
    target = anim_subject_height_px(load_spec("specs/velvet-lou.json").height_inches)
    heights = []
    for index, path in sorted(run["cells"].items()):
        pts = motion.frames[index].pts
        stretch = pose_extent(pts, motion.floor_y, motion.body_h) / stature(pts, motion.body_h)
        height = cell_height(path)[0]
        assert height == pytest.approx(target * stretch, abs=2)
        heights.append(height)
    assert len(set(heights)) > 1  # the poses really are telling them apart


def fake_sheet_draw(prompt, out_path, references=(), **kwargs):
    """One render of SHEET_FRAMES figures in a 4-wide grid, each a different shade."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    per_row = animation.FIGURES_PER_ROW
    rows = -(-animation.SHEET_FRAMES // per_row)
    # Each later sheet comes back at a different magnification, as real ones do.
    m = 1 + 0.5 * (int(out_path.stem.split("-")[1]) // animation.SHEET_FRAMES)
    image = Image.new("RGB", (int(per_row * 120 * m), int(rows * 220 * m)), (255, 0, 255))
    for n in range(animation.SHEET_FRAMES):
        x, y = int(((n % per_row) * 120 + 40) * m), int(((n // per_row) * 220 + 20) * m)
        image.paste((20 + 10 * n,) * 3, (x, y, x + int(20 * m), y + int(160 * m)))
    image.save(out_path)
    fake_sheet_draw.calls.append({"prompt": prompt, "out": out_path, "refs": list(references)})
    return out_path


@pytest.fixture
def sheet_run(tmp_path, monkeypatch):
    fake_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    graph = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_sheet_draw, sheet_mode=True
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


def test_sheet_mode_draws_the_set_in_as_few_renders_as_it_fits(sheet_run):
    assert len(fake_sheet_draw.calls) == -(-16 // animation.SHEET_FRAMES)
    assert sorted(sheet_run["sources"]) == list(range(16))
    assert sorted(sheet_run["cells"]) == list(range(16))


def test_sheet_mode_slices_figures_back_in_frame_order(sheet_run):
    # Shade rises with position on the sheet, so frame order follows reading order.
    shades = []
    for index in range(16):
        with Image.open(sheet_run["sources"][index]) as source:
            shades.append(source.getpixel((source.width // 2, source.height // 2))[0])
    assert shades == [20 + 10 * (i % animation.SHEET_FRAMES) for i in range(16)]


def test_sheet_mode_registers_every_frame_at_one_scale(sheet_run):
    """Same size on a sheet means same size in the cell — and across sheets too,
    though the fake draws the second one half again as large."""
    assert sheet_run["sheets"] == [list(range(8)), list(range(8, 16))]
    heights = set()
    for index in range(16):
        with Image.open(sheet_run["cells"][index]) as cell:
            assert cell.size == (480, 560)
            heights.add(mask.subject_box(cell)[3] - mask.subject_box(cell)[1])
    # Nearest-neighbour rounding from two source sizes; a missed sheet would be ~200px off.
    assert max(heights) - min(heights) <= 2


def test_sheet_mode_without_landmarks_measures_the_sheet_itself(tmp_path, monkeypatch):
    """No skeleton means no per-pose reading; the sheet's own figures set the scale."""
    import json

    raw = json.loads(Path(SAMPLE).read_text())
    for frame in raw["frames"]:
        frame.pop("pts", None)
    stripped = tmp_path / "motion.json"
    stripped.write_text(json.dumps(raw))
    fake_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    result = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_sheet_draw, sheet_mode=True
    ).invoke(
        {
            "spec": load_spec("specs/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,  # would put a 160px figure at 460px; the sheet must not use it
            "motion": load_motion(stripped),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    for call in fake_sheet_draw.calls:
        assert "stick-figure" not in call["prompt"]
    for index in range(16):
        with Image.open(result["cells"][index]) as cell:
            top, bottom = mask.subject_box(cell)[1], mask.subject_box(cell)[3]
            assert abs((bottom - top) - anim_subject_height_px(69)) <= 2


def test_sheet_prompt_shows_every_pose_and_measures_nothing(sheet_run):
    for call in fake_sheet_draw.calls:
        prompt = call["prompt"]
        assert f"{animation.SHEET_FRAMES} times in one image" in prompt
        assert "same size" in prompt
        assert "tall" not in prompt and "px" not in prompt and "480" not in prompt
        assert "stick-figure skeleton" in prompt
        grid = Path(call["refs"][-1])
        assert grid.parts[-3] == "poses" and grid.stem.startswith("sheet-")
        with Image.open(grid) as image:
            assert image.size == (480 * animation.FIGURES_PER_ROW, 560 * 2)


def test_a_sheet_with_the_wrong_figure_count_is_kept_and_redrawn(tmp_path, monkeypatch):
    def flaky(prompt, out_path, references=(), **kwargs):
        if not flaky.calls:
            out_path = Path(out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            image = Image.new("RGB", (400, 200), (255, 0, 255))
            image.paste((40, 40, 40), (40, 20, 60, 180))
            image.paste((80, 80, 80), (200, 20, 220, 180))
            image.save(out_path)
            flaky.calls.append(out_path)
            return out_path
        return fake_sheet_draw(prompt, out_path, references, **kwargs)

    flaky.calls = []
    fake_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    result = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=flaky, sheet_mode=True
    ).invoke(
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
    assert sorted(result["cells"]) == list(range(16))
    assert (tmp_path / "lou/source/dance/sheet-00.rejected-0.png").exists()
    assert (tmp_path / "lou/source/dance/sheet-00.png").exists()
