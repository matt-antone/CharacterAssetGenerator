import numpy as np
from PIL import Image

from cag import snap

KEY_COLOURS = [(192, 23, 35), (75, 87, 111), (185, 107, 79), (58, 42, 35), (0, 0, 0)]


def key_art(tmp_path):
    """A flat, on-grid figure in a handful of colours, on pure magenta."""
    a = np.full((96, 64, 3), (255, 0, 255), np.uint8)
    for n, colour in enumerate(KEY_COLOURS):
        a[16 + n * 12 : 28 + n * 12, 16:48] = colour
    path = tmp_path / "key.png"
    Image.fromarray(a).save(path)
    return path


def smooth_frame():
    """A pose render as it comes back: off-grid gradients, a thin inked line,
    and a backdrop drifted to violet."""
    a = np.full((96, 64, 3), (200, 0, 225), np.uint8)
    for y in range(20, 80):
        a[y, 18:46] = (180 + (y % 7), 30 + y // 4, 40)
    a[50, 18:46] = (20, 12, 10)  # one thin ink line across the jacket
    return Image.fromarray(a)


def test_the_backdrop_comes_back_exact_magenta_and_the_figure_outlined(tmp_path):
    out = np.asarray(snap.snap(smooth_frame(), snap.palette(key_art(tmp_path))))
    assert tuple(out[0, 0]) == snap.MAGENTA
    assert tuple(out[-1, -1]) == snap.MAGENTA
    # The column through the middle: magenta, then the black outline, then figure.
    column = [tuple(p) for p in out[:, 32]]
    first = next(i for i, p in enumerate(column) if p != snap.MAGENTA)
    assert column[first] == snap.OUTLINE


def test_every_pixel_lands_on_the_grid_in_the_key_art_s_colours(tmp_path):
    out = np.asarray(snap.snap(smooth_frame(), snap.palette(key_art(tmp_path))))
    blocks = out.reshape(96 // snap.BLOCK, snap.BLOCK, 64 // snap.BLOCK, snap.BLOCK, 3)
    assert (blocks == blocks[:, :1, :, :1]).all(), "every block is one flat colour"
    allowed = {snap.MAGENTA, snap.OUTLINE, *KEY_COLOURS}
    assert {tuple(p) for p in out.reshape(-1, 3)} <= allowed


def test_a_thin_inked_line_survives_the_average(tmp_path):
    out = np.asarray(snap.snap(smooth_frame(), snap.palette(key_art(tmp_path))))
    row = out[50, 24:40]
    assert (row.sum(axis=1) < 200).all(), "the line stays dark instead of averaging away"


def test_snapping_a_file_keeps_the_render_and_never_snaps_twice(tmp_path):
    key = key_art(tmp_path)
    frame = tmp_path / "07.png"
    smooth_frame().save(frame)
    snap.snap_file(frame, key)
    raw = tmp_path / "07.raw.png"
    assert raw.exists()
    first = frame.read_bytes()
    Image.new("RGB", (64, 96), "white").save(raw)  # would change the result if re-read
    snap.snap_file(frame, key)
    assert frame.read_bytes() == first
