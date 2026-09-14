from pathlib import Path

import numpy
import pytest
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
    return graph.invoke(
        {"spec": load_spec("specs/velvet-lou.json"), "work_dir": tmp_path / "lou"}
    )


def test_draws_key_art_before_the_projection_views(run):
    order = [call["out"].stem for call in fake_draw.calls]
    assert order[0] == KEY_VIEW
    assert sorted(order[1:]) == ["back", "front", "profile"]


def test_projection_views_reference_the_key_art(run):
    key_art = fake_draw.calls[0]["out"]
    assert all(call["refs"] == [key_art] for call in fake_draw.calls[1:])
    assert fake_draw.calls[0]["refs"] == []


def test_every_prompt_quotes_the_locked_bible(run):
    assert all(BIBLE in call["prompt"] for call in fake_draw.calls)


def test_produces_four_registered_cells(run):
    assert sorted(run["cells"]) == ["back", "front", "key", "profile"]
    for path in run["cells"].values():
        with Image.open(path) as cell:
            assert cell.size == (480, 560)
            assert cell.mode == "RGBA"


def test_scale_is_measured_once_from_the_key_art(run):
    # 160px of drawn subject must become 5'9" == 460px of cell.
    assert run["scale"] == pytest.approx(460 / 160)
