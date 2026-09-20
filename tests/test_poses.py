import json
from pathlib import Path

import numpy
from PIL import Image

from cag.motion import read_bundle
from cag.poses import CARD_HEIGHT, CARD_WIDTH, write_photos


def shots(tmp_path, count=4, size=(120, 160)):
    """Stand-in traced frames, each a distinct flat colour."""
    paths = []
    for index in range(count):
        path = tmp_path / f"f{index:02d}.jpg"
        Image.new("RGB", size, (20 + index * 30, 40, 60)).save(path)
        paths.append(path)
    return paths


def test_photo_cards_keep_one_scale_across_the_set(tmp_path):
    """Fitting each frame to its own card would flatten exactly the travel the
    reference is there to show."""
    wide, tall = tmp_path / "wide.jpg", tmp_path / "tall.jpg"
    Image.new("RGB", (400, 200), (90, 30, 30)).save(wide)
    Image.new("RGB", (200, 400), (30, 90, 30)).save(tall)

    cards = write_photos([wide, tall], tmp_path / "out")
    assert [c.name for c in cards] == ["00.png", "01.png"]
    widths = []
    for card in cards:
        with Image.open(card) as handle:
            assert handle.size == (CARD_WIDTH, CARD_HEIGHT)
            pixels = numpy.asarray(handle.convert("RGB"))
            widths.append(int((pixels.sum(axis=2) > 0).any(axis=0).sum()))
    # One scale for both: the source frames are 400 and 200 wide, and the cards
    # keep that 2:1. Fitting each to its own card would make them the same width.
    assert widths[0] == 2 * widths[1]


def test_photo_cards_are_never_blown_up_past_their_shipped_size(tmp_path):
    """A thumbnail smaller than the card is pasted at the size it was shipped
    at. Blowing it up adds no detail and softens the edges the pose is read
    from, and is not what the amplitude was measured on."""
    small = tmp_path / "small.jpg"
    Image.new("RGB", (80, 100), (200, 200, 200)).save(small)
    card = write_photos([small], tmp_path / "out")[0]
    with Image.open(card) as handle:
        pixels = numpy.asarray(handle.convert("RGB"))
    lit = (pixels.sum(axis=2) > 0)
    assert lit.any(axis=0).sum() == 80, "pasted at its native width, letterboxed"
    assert lit.any(axis=1).sum() == 100


def test_a_real_bundle_hands_over_one_photograph_per_frame():
    bundle = read_bundle(Path("motions/shuffle-1"))
    assert len(bundle.photos) == bundle.frame_count
    assert [p.name for p in bundle.photos] == [f"f{i:02d}.jpg" for i in range(bundle.frame_count)]
    assert bundle.load().photos == bundle.photos


def test_a_bundle_missing_a_photograph_carries_none_rather_than_mispairing(tmp_path):
    """A partial set would silently pair figure n with the wrong frame.

    There is no second pose reference to fall back to: the set is drawn from its
    cues alone, which is loud in the render rather than quietly half-sized.
    """
    source = Path("motions/shuffle-1")
    manifest = json.loads((source / "manifest.json").read_text())
    manifest["files"] = {
        name: value for name, value in manifest["files"].items() if name != "thumbs/f05.jpg"
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for name in manifest["files"]:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((source / name).read_bytes())
    assert read_bundle(tmp_path).photos == ()


def test_a_spritesheet_in_the_manifest_is_ignored():
    """MotionArtist still ships one; nothing here reads it."""
    bundle = read_bundle(Path("motions/shuffle-1"))
    assert not hasattr(bundle, "poses")
    assert not hasattr(bundle.load(), "pose_layout")
