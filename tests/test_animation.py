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


def test_keyframes_reference_only_the_key_art(run):
    key_art = fake_draw.calls[0]["refs"][0]
    for call in fake_draw.calls[:6]:
        assert call["refs"] == [key_art]


def test_inbetweens_reference_the_key_art_and_both_neighbours(run):
    by_index = {int(c["out"].stem): c for c in fake_draw.calls}
    # Frame 1 sits between locked frames 0 and 2.
    assert order([{"out": Path(p)} for p in by_index[1]["refs"][1:]]) == [0, 2]
    # Frame 8 follows locked frame 7 and runs up to pilot frame 9.
    assert order([{"out": Path(p)} for p in by_index[8]["refs"][1:]]) == [7, 9]


def test_the_last_inbetween_closes_onto_frame_zero(run):
    by_index = {int(c["out"].stem): c for c in fake_draw.calls}
    assert order([{"out": Path(p)} for p in by_index[15]["refs"][1:]]) == [14, 0]


def test_every_frame_prompt_carries_the_directors_note_and_view(run):
    for call in fake_draw.calls:
        assert NOTE in call["prompt"]
        assert "facing the viewer square-on" in call["prompt"]


def test_produces_sixteen_registered_cells(run):
    assert sorted(run["cells"]) == list(range(16))
    with Image.open(run["cells"][0]) as cell:
        assert cell.size == (480, 560)
