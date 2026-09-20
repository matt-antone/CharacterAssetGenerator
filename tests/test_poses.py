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


SAMPLE = Path("motions/sample")


def copied(tmp_path, drop=None, **manifest_overrides):
    """The sample bundle in a temp directory, optionally missing one thumb."""
    root = tmp_path / "bundle"
    (root / "thumbs").mkdir(parents=True)
    manifest = json.loads((SAMPLE / "manifest.json").read_text())
    manifest.update(manifest_overrides)
    if drop:
        del manifest["files"][drop]
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "motion.json").write_text((SAMPLE / "motion.json").read_text())
    for name in manifest["files"]:
        if name != "motion.json":
            (root / name).write_bytes((SAMPLE / name).read_bytes())
    return root


def test_the_sample_bundle_hands_over_one_photograph_per_frame():
    """motions/sample is the worked example of the input contract. If this fails,
    the contract and the loader have drifted apart."""
    bundle = read_bundle(SAMPLE)
    assert len(bundle.photos) == bundle.frame_count
    assert [p.name for p in bundle.photos] == [f"f{i:02d}.jpg" for i in range(bundle.frame_count)]
    assert bundle.load().photos == bundle.photos
    assert bundle.load().has_poses, "every frame carries landmarks"


def test_a_bundle_missing_a_photograph_carries_none_rather_than_mispairing(tmp_path):
    """A partial set would silently pair figure n with the wrong frame.

    There is no second pose reference to fall back to: the set is drawn from its
    cues alone, which is loud in the render rather than quietly half-sized.
    """
    assert read_bundle(copied(tmp_path, drop="thumbs/f02.jpg")).photos == ()


def test_an_unknown_manifest_key_is_ignored(tmp_path):
    """A bundle may ship and declare a pose grid of its own; nothing here reads
    it, and a key the loader does not know must not stop the bundle loading."""
    bundle = read_bundle(copied(
        tmp_path,
        pose_grid={"cols": 4, "rows": 2, "tile_w": 60, "tile_h": 70},
        seam_ratio=1.49,
    ))
    assert len(bundle.photos) == bundle.frame_count
    assert not hasattr(bundle, "poses")
    assert not hasattr(bundle.load(), "pose_layout")
