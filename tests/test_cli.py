"""End to end through the CLI, with the model and the image tool faked out."""

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage
from PIL import Image, ImageSequence

from cag import animation, cli, mask, static_sheet
from tests.test_animation import fake_draw
from tests.test_static_sheet import flat_cutout

SAMPLE = "/Users/matthewantone/Development/MotionArtist/work/sample/motion.json"


@pytest.fixture
def built(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(animation, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "ChatCodex",
        lambda *a, **kw: FakeMessagesListChatModel(
            responses=[AIMessage("A lounge performer."), AIMessage("Mic in the character-right hand.")]
        ),
    )
    cli.main(
        [
            "build",
            "specs/velvet-lou.json",
            "--set", "dance",
            "--motion", SAMPLE,
            "--per-frame",
            "--work", str(tmp_path / "work"),
            "--out", str(tmp_path / "out"),
        ]
    )
    return tmp_path / "out" / "velvet-lou"


def test_writes_the_four_projection_views(built):
    for view in ("key", "front", "back", "profile"):
        with Image.open(built / "views" / f"{view}.png") as cell:
            assert cell.size == (480, 560)


def test_writes_a_wrapped_sprite_sheet(built):
    with Image.open(built / "dance-sheet.png") as sheet:
        assert sheet.size == (480 * 8, 560 * 2)  # 16 frames, 8 to a row


def test_writes_a_looping_proof_at_the_declared_rate(built):
    with Image.open(built / "dance-proof.gif") as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 16
        assert gif.info["duration"] == 250
        assert gif.info["loop"] == 0


def test_writes_a_gallery_that_points_at_the_deliverables(built):
    page = (built / "index.html").read_text()
    assert "Velvet Lou" in page
    for reference in ("views/key.png", "dance-sheet.png", "dance-proof.gif", "16 frames at 4 fps"):
        assert reference in page


def test_static_only_build_skips_the_animation(tmp_path, monkeypatch):
    fake_draw.calls = []
    monkeypatch.setattr(mask, "cutout", flat_cutout)
    monkeypatch.setattr(static_sheet, "draw", fake_draw)
    monkeypatch.setattr(
        cli,
        "ChatCodex",
        lambda *a, **kw: FakeMessagesListChatModel(responses=[AIMessage("A lounge performer.")]),
    )
    monkeypatch.setattr(cli, "write_motion", lambda *a, **kw: pytest.fail("no sheet needed"))
    cli.main(
        ["build", "specs/no-animations.json", "--work", str(tmp_path / "w"),
         "--out", str(tmp_path / "o")]
    )
    out = tmp_path / "o" / "no-one"
    assert (out / "index.html").exists()
    assert not list(out.glob("*.gif"))
    assert len(fake_draw.calls) == 4
