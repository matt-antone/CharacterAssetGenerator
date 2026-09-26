"""The cell finish, on synthetic cells: what each of its four rules does and does not do."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

from cag import finish
from cag.finish import STAMP, defringe, erode, finish_cell, finish_set, set_palette

DENIM = (60, 80, 140)
FRINGE = (200, 60, 200)
POCKET = (180, 40, 190)
PALE = (240, 215, 235)
YELLOW = (230, 200, 40)


def cell(size=(40, 40)):
    return np.zeros((size[1], size[0], 4), np.uint8)


def paint(a, box, colour):
    left, top, right, bottom = box
    a[top:bottom, left:right, :3] = colour
    a[top:bottom, left:right, 3] = 255


def reference(path, *patches, size=(40, 40)):
    """A set reference: magenta backdrop, with a patch of each colour given."""
    image = np.zeros((size[1], size[0], 3), np.uint8)
    image[...] = (255, 0, 255)
    for k, colour in enumerate(patches):
        image[2 + 6 * k : 6 + 6 * k, 2:6] = colour
    Image.fromarray(image).save(path)
    return path


def fringed_figure():
    """A denim block ringed by a 1px magenta fringe, with a magenta pocket in the middle."""
    a = cell()
    paint(a, (5, 5, 35, 35), FRINGE)
    paint(a, (6, 6, 34, 34), DENIM)
    paint(a, (18, 18, 22, 22), POCKET)
    return a


def legs():
    """Two legs meeting at the crotch, the gap between them a wedge tapering to one pixel.

    The wedge's rim is tinted, as a cut-out's is where the backdrop showed through.
    """
    a = cell((40, 60))
    paint(a, (8, 5, 32, 55), DENIM)
    wedge = np.zeros(a.shape[:2], bool)
    for y in range(30, 55):
        half = (y - 30) // 4
        wedge[y, 20 - half : 21 + half] = True
    a[wedge] = 0
    grown = Image.fromarray(wedge.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(3))
    rim = (np.asarray(grown) > 0) & ~wedge
    a[rim & (a[..., 3] > 0), :3] = FRINGE
    return a, wedge


def test_fringe_on_the_silhouette_leaves_the_figure():
    out = defringe(fringed_figure())
    assert (out[5, 5:35, 3] == 0).all() and (out[5:35, 34, 3] == 0).all(), "the fringe ring is gone"
    assert (out[6:34, 6:34, 3] == 255).all(), "the figure inside it is untouched"
    assert (out[6, 6:34, :3] == DENIM).all()


def test_an_inner_pocket_takes_its_neighbours_colour():
    out = defringe(fringed_figure())
    pocket = out[18:22, 18:22]
    assert (pocket[..., 3] == 255).all(), "a pocket inside the figure stays figure"
    assert np.abs(pocket[..., :3].astype(int) - DENIM).max() <= 1, "and is filled with the denim around it"
    assert not finish.tinted(out[..., :3])[out[..., 3] > 0].any()


def test_alpha_comes_out_all_or_nothing():
    a = fringed_figure()
    a[10, 10, 3] = 100  # a soft edge pixel from the shrink
    out = finish_cell(defringe(a), None)
    assert set(np.unique(out[..., 3])) <= {0, 255}
    assert out[10, 10, 3] == 0, "under half opaque is outside"


def test_the_gap_between_the_legs_stays_open_to_its_tip(tmp_path):
    a, wedge = legs()
    ref = reference(tmp_path / "ref.png", PALE, DENIM)
    cut = defringe(a)
    palette = set_palette(ref, [cut])
    out = finish_cell(cut, palette)
    assert (out[..., 3] == cut[..., 3]).all(), "the silhouette is the cut-out's, pixel for pixel"
    assert (out[wedge, 3] == 0).all(), "not one pixel of the wedge is refilled"
    rgb = out[..., :3][out[..., 3] > 0].astype(int)
    assert not (np.abs(rgb - PALE).sum(-1) < 40).any(), "no pale pixel appears at the crotch"


def test_the_outline_is_drawn_inside_the_silhouette():
    a = cell()
    paint(a, (10, 8, 30, 34), DENIM)
    out = finish_cell(defringe(a), None)
    fig = a[..., 3] > 0
    assert ((out[..., 3] > 0) == fig).all(), "the figure grows by nothing"
    edge = fig & ~erode(fig)
    assert (out[edge, :3] == 0).all(), "its own edge pixels are the black outline"
    inside = erode(fig)
    assert (out[inside, :3] == DENIM).all(), "and only those"


def test_the_palette_is_the_reference_and_the_sets_own_pixels(tmp_path):
    ref = reference(tmp_path / "ref.png", YELLOW)
    a = cell()
    paint(a, (10, 10, 30, 30), DENIM)
    palette = set_palette(ref, [a])

    def nearest(colour):
        probe = Image.new("RGB", (1, 1), colour).quantize(palette=palette, dither=Image.Dither.NONE)
        return probe.convert("RGB").getpixel((0, 0))

    assert nearest(YELLOW) == YELLOW, "the set reference's figure is in it"
    assert nearest(DENIM) == DENIM, "the set's own colours are too, so denim is not greyed"
    assert nearest((255, 0, 255)) != (255, 0, 255), "the backdrop is not"


def test_a_set_with_no_figure_anywhere_is_left_as_it_is(tmp_path):
    ref = reference(tmp_path / "ref.png")
    assert set_palette(ref, [cell()]) is None
    assert (finish_cell(cell(), None) == 0).all()


def cuts_on_disk(tmp_path, n=3):
    folder = tmp_path / "cells-cut" / "dance"
    folder.mkdir(parents=True)
    cuts = {}
    for index in range(n):
        a = fringed_figure()
        paint(a, (10, 10 + index, 14, 14 + index), YELLOW)
        cuts[index] = folder / f"{index:02d}.png"
        Image.fromarray(a).save(cuts[index])
    return cuts


def test_a_set_is_finished_once_and_then_left_alone(tmp_path, capsys):
    cuts = cuts_on_disk(tmp_path)
    ref = reference(tmp_path / "ref.png", DENIM)
    dst = lambda index: tmp_path / "cells" / "dance" / f"{index:02d}.png"  # noqa: E731
    cells = finish_set(cuts, dst, ref, "dance")
    assert cells == {index: dst(index) for index in cuts}
    assert (tmp_path / "cells" / "dance" / STAMP).exists()
    first = {index: path.read_bytes() for index, path in cells.items()}
    times = {index: path.stat().st_mtime_ns for index, path in cells.items()}
    for path in cuts.values():
        with Image.open(path) as image:
            assert 128 < np.asarray(image)[20, 5, 3], "the cut-outs are kept as cut"

    # The same cut-outs, cut again byte for byte, finish nothing.
    for path in cuts.values():
        path.write_bytes(path.read_bytes())
    finish_set(cuts, dst, ref, "dance")
    assert {index: path.stat().st_mtime_ns for index, path in cells.items()} == times
    assert "cell finish cached" in capsys.readouterr().err

    # A changed set reference redoes it, from the cut-outs, to the same cells.
    reference(ref, DENIM, YELLOW)
    finish_set(cuts, dst, ref, "dance")
    assert "cell finish cached" not in capsys.readouterr().err
    reference(ref, DENIM)
    finish_set(cuts, dst, ref, "dance")
    assert {index: path.read_bytes() for index, path in cells.items()} == first, (
        "never finished on top of its own output"
    )


def test_a_changed_cut_out_or_a_lost_cell_redoes_the_set(tmp_path, capsys):
    cuts = cuts_on_disk(tmp_path)
    ref = reference(tmp_path / "ref.png", DENIM)
    dst = lambda index: tmp_path / "cells" / "dance" / f"{index:02d}.png"  # noqa: E731
    finish_set(cuts, dst, ref, "dance")
    capsys.readouterr()

    a = fringed_figure()
    paint(a, (20, 25, 24, 29), YELLOW)
    Image.fromarray(a).save(cuts[1])
    finish_set(cuts, dst, ref, "dance")
    assert "cell finish cached" not in capsys.readouterr().err

    dst(2).unlink()
    finish_set(cuts, dst, ref, "dance")
    assert "cell finish cached" not in capsys.readouterr().err and dst(2).exists()


def test_a_new_finish_version_redoes_every_set(tmp_path, monkeypatch, capsys):
    cuts = cuts_on_disk(tmp_path)
    ref = reference(tmp_path / "ref.png", DENIM)
    dst = lambda index: tmp_path / "cells" / "dance" / f"{index:02d}.png"  # noqa: E731
    finish_set(cuts, dst, ref, "dance")
    capsys.readouterr()
    monkeypatch.setattr(finish, "FINISH_VERSION", "finish/test")
    finish_set(cuts, dst, ref, "dance")
    assert "cell finish cached" not in capsys.readouterr().err
    assert Path(tmp_path / "cells" / "dance" / STAMP).read_text().strip() == finish.finish_key(cuts, ref)
