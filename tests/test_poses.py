import json
from pathlib import Path

import pytest
from PIL import Image

from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from cag.motion import read_bundle
from cag.poses import PoseSheetError, SheetLayout, cut, write_poses

BLOCK = {
    "cols": 4, "rows": 3, "scale": 2,
    "tile_w": 30, "tile_h": 40, "cell_w": 25, "cell_h": 30, "label_h": 5,
}
#: Every tile a different shade, so a frame that reads the wrong tile is caught
#: rather than merely looking plausible.
SHADES = [(10 * i + 5, 40, 200 - 4 * i) for i in range(12)]


@pytest.fixture
def sheet(tmp_path):
    layout = SheetLayout.from_manifest(BLOCK)
    image = Image.new("RGB", (4 * 60, 3 * 80), (0, 0, 0))
    for index, shade in enumerate(SHADES):
        left, top, right, bottom = layout.box(index)
        image.paste(shade, (left, top, right, bottom))
        # A band above each figure carries the frame number; it must not be cut.
        image.paste((255, 255, 255), (left, top - int(layout.label_h), right, top))
    path = tmp_path / "poses.png"
    image.save(path)
    return path, layout


def test_each_frame_reads_its_own_tile_in_reading_order(sheet):
    path, layout = sheet
    cells = cut(path, layout, 12)
    assert [cell.getpixel((1, 1)) for cell in cells] == SHADES


def test_the_frame_number_band_is_left_behind(sheet):
    """The label sits above the figure; cropping below it means the generator
    never sees a number it could copy into the art."""
    path, layout = sheet
    for cell in cut(path, layout, 12):
        colours = {colour for _, colour in cell.convert("RGB").getcolors(maxcolors=1 << 16)}
        assert (255, 255, 255) not in colours


def test_scale_converts_the_declared_units_to_pixels():
    layout = SheetLayout.from_manifest(BLOCK)
    assert (layout.tile_w, layout.cell_h, layout.label_h) == (60, 60, 10)
    assert layout.box(5) == (60, 90, 110, 150)  # row 1, column 1


def test_a_grid_too_small_for_the_set_is_refused(sheet):
    path, layout = sheet
    with pytest.raises(PoseSheetError, match="too small"):
        cut(path, layout, 13)


def test_a_missing_measurement_is_refused_rather_than_guessed():
    with pytest.raises(PoseSheetError, match="cell_w"):
        SheetLayout.from_manifest({k: v for k, v in BLOCK.items() if k != "cell_w"})


def test_poses_are_written_cell_sized_and_all_at_one_scale(sheet, tmp_path):
    """One factor for the whole set: the sheet draws every frame at a common
    scale, and fitting each frame to the cell on its own would throw that away."""
    path, layout = sheet
    written = write_poses(path, layout, 12, tmp_path / "out")
    assert [p.name for p in written] == [f"{i:02d}.png" for i in range(12)]
    boxes = []
    for each in written:
        with Image.open(each) as cell:
            assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
            boxes.append(cell.getbbox())
    assert len(set(boxes)) == 1


def test_a_real_bundle_hands_over_its_sprite_sheet():
    bundle = read_bundle("motions/shuffle")
    motion = bundle.load()
    assert motion.poses and motion.poses.exists()
    assert motion.pose_layout.columns * motion.pose_layout.rows >= len(motion.frames)
    assert len(cut(motion.poses, motion.pose_layout, len(motion.frames))) == len(motion.frames)


def test_a_bundle_without_a_sprite_sheet_carries_no_poses(tmp_path):
    """An older export has no sheet. The set still builds; it just gets no
    pose reference, rather than one cag drew for itself."""
    source = Path("motions/shuffle")
    root = tmp_path / "shuffle"
    root.mkdir()
    (root / "motion.json").write_text((source / "motion.json").read_text())
    manifest = json.loads((source / "manifest.json").read_text())
    manifest.pop("spritesheet")
    manifest["files"] = {"motion.json": ""}
    (root / "manifest.json").write_text(json.dumps(manifest))

    bundle = read_bundle(root)
    assert bundle.poses is None
    assert bundle.load().pose_layout is None
