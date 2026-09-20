import json
from pathlib import Path

import numpy
import pytest

from cag.geometry import CELL_HEIGHT, CELL_WIDTH, subject_height_px
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image

from cag import mask, static_sheet
from cag.prompts import KEY_VIEW
from cag.spec import load_spec

BIBLE = "A lounge performer in a green velvet jacket, microphone in the character-right hand."


def fake_draw(prompt, out_path, references=(), **kwargs):
    """Draw a figure whose size encodes nothing but keeps the mask path honest."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        return out_path  # like the real draw: a render on disk is never redrawn
    image = Image.new("RGBA", (100, 200), (255, 0, 255, 255))
    image.paste((20, 20, 20, 255), (40, 20, 60, 180))
    image.save(out_path)
    fake_draw.calls.append({"prompt": prompt, "out": out_path, "refs": list(references)})
    return out_path


def flat_cutout(src):
    """Stand in for Vision: the fakes are flat PNGs, so key out the magenta."""
    pixels = numpy.array(Image.open(src).convert("RGBA"))
    background = (pixels[:, :, 0] > 200) & (pixels[:, :, 1] < 60) & (pixels[:, :, 2] > 200)
    pixels[background] = 0
    return Image.fromarray(pixels, "RGBA")


@pytest.fixture
def run(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(static_sheet, "cutout", flat_cutout)
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    model = FakeMessagesListChatModel(responses=[AIMessage(BIBLE)])
    graph = static_sheet.build_static_graph(model, draw_fn=fake_draw)
    state = {"spec": load_spec("tests/fixtures/velvet-lou.json"), "work_dir": tmp_path / "lou"}
    # The first pass stops at the key art gate; sign off, then draw the rest.
    with pytest.raises(static_sheet.ApprovalRequired):
        graph.invoke(state)
    static_sheet.approve(tmp_path / "lou")
    return graph.invoke(state)


def test_nothing_but_the_key_art_is_drawn_before_approval(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(static_sheet, "cutout", flat_cutout)
    model = FakeMessagesListChatModel(responses=[AIMessage(BIBLE)])
    graph = static_sheet.build_static_graph(model, draw_fn=fake_draw)
    with pytest.raises(static_sheet.ApprovalRequired):
        graph.invoke({"spec": load_spec("tests/fixtures/velvet-lou.json"), "work_dir": tmp_path / "lou"})
    assert [call["out"].stem for call in fake_draw.calls] == [KEY_VIEW]


def test_approval_does_not_carry_over_to_redrawn_key_art(tmp_path):
    work_dir = tmp_path / "lou"
    key_art = static_sheet.source_path(work_dir, KEY_VIEW)
    fake_draw("", key_art)
    static_sheet.approve(work_dir)
    assert static_sheet.is_approved(work_dir, key_art)

    Image.new("RGBA", (100, 200), (255, 0, 255, 255)).save(key_art)  # a different render
    assert not static_sheet.is_approved(work_dir, key_art)


def test_draws_key_art_before_the_projection_views(run):
    order = [call["out"].stem for call in fake_draw.calls]
    assert order[0] == KEY_VIEW
    assert sorted(order[1:]) == ["back", "front", "profile"]


def test_projection_views_reference_the_key_art(run):
    key_art = fake_draw.calls[0]["out"]
    assert all(call["refs"][0] == key_art for call in fake_draw.calls[1:])


def test_every_view_names_its_detail_level_and_the_arcade_style(run):
    from cag.style import DEFAULT_DETAIL_LEVEL, detail_frame

    for call in fake_draw.calls:
        assert f"detail level {DEFAULT_DETAIL_LEVEL}" in call["prompt"]
        assert "32-bit arcade sprite art" in call["prompt"]
        assert "character-left and character-right" in call["prompt"]


def test_the_detail_sample_rides_along_only_when_the_level_has_one(run):
    """Only level 4 ships a frame; other levels get the words alone."""
    from cag.style import DEFAULT_DETAIL_LEVEL, detail_frame

    sample = detail_frame(DEFAULT_DETAIL_LEVEL)
    for call in fake_draw.calls:
        names = [Path(p).name for p in call["refs"]]
        assert (sample.name in names) if sample else (not any("detail-level" in n for n in names))


def test_every_prompt_quotes_the_locked_bible(run):
    assert all(BIBLE in call["prompt"] for call in fake_draw.calls)


def test_produces_four_registered_cells(run):
    assert sorted(run["cells"]) == ["back", "front", "key", "profile"]
    for path in run["cells"].values():
        with Image.open(path) as cell:
            assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
            assert cell.mode == "RGBA"


def test_scale_is_measured_once_from_the_key_art(run):
    # 160px of drawn subject must become 5'9" at the cell's own scale.
    assert run["scale"] == pytest.approx(subject_height_px(69) / 160)


def test_a_bible_on_disk_is_reused_rather_than_rewritten(tmp_path):
    """Later sets must quote the identity the earlier frames were drawn against."""
    work = tmp_path / "lou"
    work.mkdir()
    (work / "bible.txt").write_text(BIBLE + "\n")
    model = FakeMessagesListChatModel(responses=[AIMessage("A different performer entirely.")])
    state = static_sheet.write_bible({"spec": load_spec("tests/fixtures/velvet-lou.json"), "work_dir": work}, model)
    assert state["bible"] == BIBLE


def test_the_bible_writer_is_told_the_costume_the_brief_states(tmp_path):
    """The boots are in `outfit`, so the boots reach the paragraph.

    Before the brief had fields, a costume fact only reached the bible if the
    designer happened to write it into the description blob, and only reached a
    prompt if a regex found it again afterwards.
    """
    from cag.prompts import bible_request
    from cag.spec import load_spec as load

    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Velvet Lou", "height": "5' 9\"", "description": "A lounge performer.",
        "outfit": "Scuffed brown boots.", "avoid": ["trainers", "a hat"],
        "prop": "A chrome microphone.", "personality": "Cheerful.",
        "animations": {"dance": "A two-step loop."},
    }))
    request = bible_request(load(brief))
    assert "Outfit: Scuffed brown boots." in request
    assert "Avoid: trainers; a hat" in request
    # BIBLE_SYSTEM forbids both: a prop named here would follow the character
    # into every set, and personality is not visible.
    assert "chrome microphone" not in request
    assert "Cheerful" not in request


def test_a_brief_with_no_fields_asks_for_the_bible_exactly_as_it_always_did():
    from cag.prompts import bible_request
    from cag.spec import load_spec as load

    spec = load("tests/fixtures/velvet-lou.json")
    assert bible_request(spec) == (
        f"Character: {spec.name}\nHeight: {spec.height}\nDesigner's brief: {spec.description}"
    )


def test_a_brief_that_states_its_appearance_needs_no_model(tmp_path):
    """And no freeze: the text is a function of the spec, so the spec is the record."""
    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Velvet Lou", "height": "5' 9\"", "description": "A lounge performer.",
        "build": "Lean.", "outfit": "Scuffed brown boots.", "palette": ["velvet #4A1E3C"],
        "prop": "A chrome microphone.", "avoid": ["trainers"],
        "animations": {"dance": "A two-step loop."},
    }))
    work = tmp_path / "lou"
    work.mkdir()
    (work / "bible.txt").write_text("A stale paragraph a model wrote once.\n")
    model = FakeMessagesListChatModel(responses=[AIMessage("Another one entirely.")])

    bible = static_sheet.write_bible({"spec": load_spec(brief), "work_dir": work}, model)["bible"]
    assert "Scuffed brown boots." in bible
    assert "velvet #4A1E3C" in bible
    assert "stale paragraph" not in bible, "the spec outranks a cached model call"
    # The leak this closes: every bible a model wrote for the roster named the
    # microphone, and that sentence was quoted into sets drawn empty-handed.
    assert "microphone" not in bible
    assert "trainers" not in bible, "an avoid list in a render prompt draws the thing"
