import re
from dataclasses import replace
from pathlib import Path

import pytest

from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from cag.poses import CARD_HEIGHT, CARD_WIDTH
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from cag import animation, mask
from cag.geometry import ANIM_CONTACT_ROW, anim_subject_height_px
from cag.motion import load_motion
from cag.mask import pose_extent, stature
from cag.prompts import CARRY_REFERENCE, EMPTY_HANDS
from cag.spec import load_spec
from tests.test_static_sheet import flat_cutout

SAMPLE = "motions/sample/motion.json"
NOTE = "The microphone stays in the character-right hand for every frame."


def posed(motion, tmp_path):
    """The sample motion with one traced frame per motion frame, the way a
    bundle carries them.

    The frames are just flat tiles: nothing here reads what is in a photograph,
    only that one card per frame is written and handed over.
    """
    shots = []
    thumbs = tmp_path / "thumbs"
    thumbs.mkdir(parents=True, exist_ok=True)
    for index in range(len(motion.frames)):
        path = thumbs / f"f{index:02d}.jpg"
        Image.new("RGB", (120, 160), (20 + index * 4, 40, 60)).save(path)
        shots.append(path)
    return replace(motion, photos=tuple(shots))


def fake_draw(prompt, out_path, references=(), **kwargs):
    """Each render differs, so GIF assembly cannot collapse identical frames."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path  # like the real draw: a render on disk is never redrawn
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
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw, frame_sheet_mode=False
    )
    return graph.invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": posed(load_motion(SAMPLE), tmp_path),
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


def test_the_pose_card_is_always_the_last_reference(run):
    for call in fake_draw.calls:
        assert pose_ref(call).parts[-3] == "poses"
        assert pose_ref(call).stem == call["out"].stem
        assert "photograph of a real performer holding this exact pose" in call["prompt"]


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
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_draw, frame_sheet_mode=False
    ).invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": load_motion(stripped),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    assert len(fake_draw.calls) == 16
    assert all("figure holding this exact pose" not in c["prompt"] for c in fake_draw.calls)


def test_every_frame_prompt_carries_the_directors_note_and_view(run):
    for call in fake_draw.calls:
        assert NOTE in call["prompt"]
        assert "facing the viewer square-on" in call["prompt"]


def test_produces_sixteen_registered_cells(run):
    assert sorted(run["cells"]) == list(range(16))
    with Image.open(run["cells"][0]) as cell:
        assert cell.size == (CELL_WIDTH, CELL_HEIGHT)


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
    target = anim_subject_height_px(load_spec("tests/fixtures/velvet-lou.json").height_inches)
    heights = []
    for index, path in sorted(run["cells"].items()):
        pts = motion.frames[index].pts
        stretch = pose_extent(pts, motion.floor_y, motion.body_h) / stature(pts, motion.body_h)
        height = cell_height(path)[0]
        assert height == pytest.approx(target * stretch, abs=2)
        heights.append(height)
    assert len(set(heights)) > 1  # the poses really are telling them apart


def fake_frame_sheet_draw(prompt, out_path, references=(), **kwargs):
    """One render of the figures the prompt asked for, in a 4-wide grid, each a different shade.

    The count comes from the prompt, not from FRAME_SHEET_SIZE: a set that does not
    divide evenly ends on a short chunk, and a stand-in that always drew a full
    sheet would hand that chunk more figures than it asked for.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    asked = int(re.search(r" (\d+) times in one image", prompt).group(1))
    per_row = min(animation.FIGURES_PER_ROW, asked)
    rows = -(-asked // per_row)
    # Each later sheet comes back at a different magnification, as real ones do.
    m = 1 + 0.5 * (int(out_path.stem.split("-")[1]) // animation.FRAME_SHEET_SIZE)
    image = Image.new("RGB", (int(per_row * 120 * m), int(rows * 220 * m)), (255, 0, 255))
    for n in range(asked):
        x, y = int(((n % per_row) * 120 + 40) * m), int(((n // per_row) * 220 + 20) * m)
        image.paste((20 + 10 * n,) * 3, (x, y, x + int(20 * m), y + int(160 * m)))
    image.save(out_path)
    fake_frame_sheet_draw.calls.append({"prompt": prompt, "out": out_path, "refs": list(references)})
    return out_path


@pytest.fixture
def sheet_run(tmp_path, monkeypatch):
    fake_frame_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    graph = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_frame_sheet_draw, frame_sheet_mode=True
    )
    return graph.invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": posed(load_motion(SAMPLE), tmp_path),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )


def test_sheet_mode_draws_the_set_in_as_few_renders_as_it_fits(sheet_run):
    assert len(fake_frame_sheet_draw.calls) == -(-16 // animation.FRAME_SHEET_SIZE)
    assert sorted(sheet_run["sources"]) == list(range(16))
    assert sorted(sheet_run["cells"]) == list(range(16))


def test_sheet_mode_slices_figures_back_in_frame_order(sheet_run):
    # Shade rises with position on the sheet, so frame order follows reading order.
    shades = []
    for index in range(16):
        with Image.open(sheet_run["sources"][index]) as source:
            shades.append(source.getpixel((source.width // 2, source.height // 2))[0])
    assert shades == [20 + 10 * (i % animation.FRAME_SHEET_SIZE) for i in range(16)]


def test_each_sheet_continues_from_the_last_figure_of_the_one_before(sheet_run):
    """Sheets are independent renders of one set, which is where costume wanders.
    Every sheet but the first is shown the frame drawn immediately before it, and
    is told so — behind the key art and never in the pose grid's last slot."""
    first, second = fake_frame_sheet_draw.calls
    carry = sheet_run["sources"][animation.FRAME_SHEET_SIZE - 1]
    assert carry not in first["refs"] and CARRY_REFERENCE not in first["prompt"]
    assert second["refs"][1] == carry and second["refs"][-1] != carry
    assert CARRY_REFERENCE in second["prompt"]


def test_the_first_sheet_continues_from_the_set_before_it(tmp_path, monkeypatch):
    """A build chains its sets, so a set is handed the last frame of the one before
    it and its opening sheet continues from that instead of from the key art alone."""
    fake_frame_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art, before = tmp_path / "key.png", tmp_path / "before.png"
    for path in (key_art, before):
        Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(path)
    animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]),
        draw_fn=fake_frame_sheet_draw,
        frame_sheet_mode=True,
    ).invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "carry": before,
            "scale": 2.875,
            "motion": posed(load_motion(SAMPLE), tmp_path),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    first = fake_frame_sheet_draw.calls[0]
    assert first["refs"][1] == before and CARRY_REFERENCE in first["prompt"]


def test_sheet_mode_registers_every_frame_at_one_scale(sheet_run):
    """Same size on a sheet means same size in the cell — and across sheets too,
    though the fake draws the second one half again as large."""
    n = animation.FRAME_SHEET_SIZE
    assert sheet_run["frame_sheets"] == [list(range(0, min(n, 16))), list(range(min(n, 16), 16))]
    heights = set()
    for index in range(16):
        with Image.open(sheet_run["cells"][index]) as cell:
            assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
            heights.add(mask.subject_box(cell)[3] - mask.subject_box(cell)[1])
    # Nearest-neighbour rounding from two source sizes, and the two sheets are
    # different lengths as well as different magnifications, so the rounding does
    # not cancel. A missed sheet would be ~200px off, not a few.
    assert max(heights) - min(heights) <= 5


def test_sheet_mode_without_landmarks_measures_the_sheet_itself(tmp_path, monkeypatch):
    """No skeleton means no per-pose reading; the sheet's own figures set the scale."""
    import json

    raw = json.loads(Path(SAMPLE).read_text())
    for frame in raw["frames"]:
        frame.pop("pts", None)
    stripped = tmp_path / "motion.json"
    stripped.write_text(json.dumps(raw))
    fake_frame_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    result = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=fake_frame_sheet_draw, frame_sheet_mode=True
    ).invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,  # would put a 160px figure at 460px; the sheet must not use it
            "motion": load_motion(stripped),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    for call in fake_frame_sheet_draw.calls:
        assert "dark card" not in call["prompt"]
    for index in range(16):
        with Image.open(result["cells"][index]) as cell:
            top, bottom = mask.subject_box(cell)[1], mask.subject_box(cell)[3]
            assert abs((bottom - top) - anim_subject_height_px(69)) <= 2


def test_sheet_prompt_shows_every_pose_and_measures_nothing(sheet_run):
    # A set that does not divide evenly ends on a short chunk, and that render is
    # asked for its own figure count, not for a full sheet.
    n = animation.FRAME_SHEET_SIZE
    expected = [min(n, 16 - start) for start in range(0, 16, n)]
    assert [len(s) for s in sheet_run["frame_sheets"]] == expected
    for call, size in zip(fake_frame_sheet_draw.calls, expected):
        prompt = call["prompt"]
        assert f"{size} times in one image" in prompt
        rows = -(-size // animation.FIGURES_PER_ROW)
        assert "same size" in prompt
        # The grid is told as cells, not as figures with gaps between them: figures
        # asked to stand apart still reach into each other, and then a box cut round
        # one carries the neighbour's hand into its frame.
        columns = min(animation.FIGURES_PER_ROW, size)
        assert f"grid of {columns} columns and {rows} row" in prompt
        assert f"{size} equal rectangular cells" in prompt
        assert "entirely within its own cell" in prompt
        # Still no measurement: no size for the figures and no ratio for the canvas.
        assert "px" not in prompt and str(CELL_WIDTH) not in prompt
        assert "pixels tall" not in prompt
        assert "a strip of photographs of a real performer, one per figure" in prompt
        grid = Path(call["refs"][-1])
        assert grid.parts[-3] == "poses" and grid.stem.startswith("pose-grid-")
        with Image.open(grid) as image:
            # Card for card, the same grid the figures are asked for, so pose N sits
            # where figure N is drawn.
            assert image.size == (CARD_WIDTH * animation.FIGURES_PER_ROW, CARD_HEIGHT * rows)


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
        return fake_frame_sheet_draw(prompt, out_path, references, **kwargs)

    flaky.calls = []
    fake_frame_sheet_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    key_art = tmp_path / "key.png"
    Image.new("RGBA", (10, 10), (255, 0, 255, 255)).save(key_art)
    result = animation.build_animation_graph(
        FakeMessagesListChatModel(responses=[AIMessage(NOTE)]), draw_fn=flaky, frame_sheet_mode=True
    ).invoke(
        {
            "spec": load_spec("tests/fixtures/velvet-lou.json"),
            "bible": "A lounge performer.",
            "key_art": key_art,
            "scale": 2.875,
            "motion": posed(load_motion(SAMPLE), tmp_path),
            "set_name": "dance",
            "work_dir": tmp_path / "lou",
        }
    )
    assert sorted(result["cells"]) == list(range(16))
    sheets = tmp_path / "lou/source/dance"
    assert list(sheets.glob("sheet-00-*.rejected-0.png"))
    assert list(sheets.glob("sheet-00-*.png"))


def test_a_different_motion_sheet_redraws_instead_of_reusing_the_old_art(sheet_run, tmp_path):
    """Reuse keys on the sheet, not the frame numbers.

    A traced sheet swapped in for a written one shares frame indices with it, so
    an index-only cache kept every render drawn from the prose it replaced.
    """
    before = len(fake_frame_sheet_draw.calls)
    # The same photographs sheet_run drew from: they are part of what "same" means.
    swapped = posed(load_motion(SAMPLE), tmp_path)
    moved = tuple(
        replace(frame, cue=f"{frame.cue} arms overhead") for frame in swapped.frames
    )
    state = {
        "spec": load_spec("tests/fixtures/velvet-lou.json"),
        "bible": "A lounge performer.",
        "key_art": tmp_path / "key.png",
        "scale": 2.875,
        "set_name": "dance",
        "work_dir": tmp_path / "lou",
        "set_note": NOTE,
        "poses": {},
    }

    animation.frame_sheet({**state, "motion": swapped}, draw_fn=fake_frame_sheet_draw)
    assert len(fake_frame_sheet_draw.calls) == before, "the same sheet must not redraw"

    animation.frame_sheet({**state, "motion": replace(swapped, frames=moved)}, draw_fn=fake_frame_sheet_draw)
    assert len(fake_frame_sheet_draw.calls) > before, "a changed sheet must redraw"


def test_new_pose_photos_with_the_same_cues_change_the_stamp(tmp_path):
    """A bundle re-cut with new photographs keeps its cues, and on the photo
    route the photographs are the whole pose reference. The stamp has to see
    them, or the frames drawn from the old photographs are kept."""
    import hashlib

    motion = load_motion(SAMPLE)
    before = animation.motion_digest(posed(motion, tmp_path / "old"))
    recut = posed(motion, tmp_path / "new")
    Image.new("RGB", (384, 512), (200, 10, 10)).save(recut.photos[0])
    assert animation.motion_digest(recut) != before

    # A set with no photographs keeps the stamp it was drawn under.
    frames = "\n".join(f"{f.index}\t{f.role}\t{f.cue}\t{f.note}" for f in motion.frames)
    unchanged = hashlib.sha256(f"{motion.view}\n{frames}".encode()).hexdigest()
    assert animation.motion_digest(motion) == unchanged


def test_the_pose_reference_is_described_as_a_photograph():
    """A photograph carries a whole person, so the prompt has to draw the line:
    everything about the pose, nothing about who is holding it. It is the only
    kind of pose reference left — the drawn cards, and the clause that described
    their teal and violet limbs, went with the sprite-sheet route."""
    from cag.prompts import FRAME_VIEWS, frame_sheet_prompt

    spec = load_spec("tests/fixtures/velvet-lou.json")
    cues = [("key", "a cue"), ("inbetween", "another")]
    shot = frame_sheet_prompt(spec, "B", "N", FRAME_VIEWS["front"], cues, pose_reference=True)

    assert "photographs of a real performer" in shot
    assert "teal green" not in shot, "the drawn-card clause is gone"
    # The two things measurement said the photo route needs: full-size movement,
    # and the character's own footwear kept on a foot that has left the floor.
    assert "not a smaller, more cautious version" in shot
    assert "including on a foot that is off the floor" in shot


def test_a_photographic_set_sends_the_lean_prompt():
    """A photograph already says what the bible, the director's note and the cues
    approximate, and repeating it in prose costs movement: the same cards drew
    0.43-0.70 of the traced amplitude carrying all three and 0.75-1.21 without.
    What a photograph cannot say is whose costume survives it, so that stays."""
    from cag.prompts import FRAME_VIEWS, frame_sheet_prompt

    spec = load_spec("tests/fixtures/velvet-lou.json")
    bible = "He is tall. Chunky ankle boots in brown leather. A tall dark quiff."
    cues = [("key", "weight centred over both feet"), ("inbetween", "knee lifted")]
    args = (spec, bible, "THE DIRECTOR NOTE", FRAME_VIEWS["front"], cues)

    # The two prompts the pipeline actually builds: a set driven by photographs,
    # and a written set with no pose reference at all.
    lean = frame_sheet_prompt(*args, pose_reference=True, photographic=True)
    full = frame_sheet_prompt(*args)

    for dropped in ("He is tall", "THE DIRECTOR NOTE", "weight centred over both feet"):
        assert dropped in full
        assert dropped not in lean
    assert "ankle boots" in lean and "quiff" in lean


def test_the_director_is_told_what_the_set_holds_rather_than_left_to_guess(tmp_path):
    """DIRECTOR_SYSTEM asks what the character holds, so silence is not an answer.

    Asked with no prop information, the director answered from whatever the
    identity text mentioned. That is how "Keep one wireless handheld microphone
    in the character-right hand" became standing instruction for dance, ko and
    victory on four characters the brief draws empty-handed in all three.
    """
    import json

    class Recorder:
        def __init__(self):
            self.seen = []

        def invoke(self, messages, *args, **kwargs):
            self.seen.append(messages[-1].content)
            return AIMessage(NOTE)

    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Velvet Lou", "height": "5' 9\"", "description": "A lounge performer.",
        "props": ["mic"],
        "animations": {"sing": {"intent": "A held note.", "props": ["mic"]},
                       "dance": "A club loop."},
    }))
    spec, motion = load_spec(brief), load_motion(SAMPLE)

    def ask(set_name):
        model = Recorder()
        animation.direct({"spec": spec, "bible": "A lounge performer.", "motion": motion,
                          "set_name": set_name, "work_dir": tmp_path / "lou"}, model)
        return model.seen[0]

    empty = ask("dance")
    assert "Both of their hands are empty" in empty
    assert "no microphone" in empty

    held = ask("sing")
    assert "Both of their hands are empty" not in held
    assert "mic" in held.lower(), "the set that does hold one still says so"


def test_a_set_without_props_sends_the_empty_hands_instruction():
    """Saying nothing let the model draw a microphone back in.

    The director and the per-set key art already sent EMPTY_HANDS; the frames
    did not, so a traced hand rising near the face came back holding a mic.
    """
    spec = load_spec("specs/default/diva.json")
    state = {"spec": replace(spec, animation_props={}), "set_name": "dance"}
    assert animation.prop_clause(state) == EMPTY_HANDS


def test_a_set_with_props_still_names_them():
    spec = load_spec("specs/default/diva.json")
    state = {"spec": replace(spec, animation_props={"dance": ("mic",)}), "set_name": "dance"}
    clause = animation.prop_clause(state)
    assert "microphone" in clause.lower()
    assert clause != EMPTY_HANDS
