import numpy as np
from PIL import Image, ImageDraw

from dataclasses import replace

from cag.mannequin import MAGENTA, Look, detail, draw_set, fit, is_flat
from cag.motion import load_motion

SAMPLE = "motions/sample/motion.json"

LOOK = Look(
    jacket=(0xC0, 0x17, 0x23), sleeve=(0xDC, 0x30, 0x3A), top=(0x22, 0x1C, 0x1E), legs=(0x4B, 0x57, 0x6F),
    shins=(0x4B, 0x57, 0x6F), boots=(0x3A, 0x2A, 0x23), skin=(0xB9, 0x6B, 0x4F), hands=(0xB9, 0x6B, 0x4F),
    cuffs=(0xB9, 0x6B, 0x4F), hair=(0x59, 0x30, 0x29), belt=(0x7A, 0x0D, 0x17),
)


def figure(image):
    a = np.asarray(image).astype(int)
    return ~np.all(a == MAGENTA, axis=-1)


def test_one_mannequin_per_traced_frame_on_the_canvas():
    sheet = load_motion(SAMPLE)
    frames = draw_set(sheet, LOOK, (512, 768))
    assert len(frames) == len(sheet.frames)
    assert all(f.size == (512, 768) for f in frames)


def test_every_frame_stays_inside_the_canvas_with_a_backdrop_margin():
    for frame in draw_set(load_motion(SAMPLE), LOOK, (512, 768)):
        fig = figure(frame)
        assert fig.any()
        assert not fig[0].any() and not fig[:, 0].any() and not fig[:, -1].any()


def test_one_fit_for_the_whole_set_so_traced_travel_survives():
    sheet = load_motion(SAMPLE)
    xf = fit(sheet, (512, 768))
    # the same landmark position maps to the same pixel whichever frame it is in
    assert xf([1.0, sheet.floor_y]) == xf([1.0, sheet.floor_y])
    assert abs(xf([0, sheet.floor_y])[1] - 768 * 0.947) < 1e-6


def test_a_reach_above_the_head_shrinks_the_set_instead_of_leaving_the_canvas():
    sheet = load_motion(SAMPLE)
    raised = replace(sheet.frames[0], pts={**sheet.frames[0].pts, "wrL": [0.5, -0.4, 0.0]})
    sheet = replace(sheet, frames=[raised, *sheet.frames[1:]])
    y = fit(sheet, (512, 768))([0.5, -0.4])[1]
    assert y >= 0.02 * 768 - 1e-6


def test_the_mannequin_is_painted_in_the_looks_colours_only():
    frame = draw_set(load_motion(SAMPLE), LOOK, (512, 768))[0]
    colours = {tuple(c) for c in np.asarray(frame).reshape(-1, 3)}
    allowed = {MAGENTA, (0x1C, 0x13, 0x10), *(v for v in vars(LOOK).values() if isinstance(v, tuple))}
    assert colours <= allowed


def test_a_flat_frame_is_told_apart_from_a_drawn_one():
    reference = Image.new("RGB", (200, 300), MAGENTA)
    d = ImageDraw.Draw(reference)
    d.rectangle([50, 50, 150, 250], fill=(120, 60, 40))
    for y in range(60, 240, 6):  # banded shading and seams: detail
        d.line([55, y, 145, y], fill=(40, 20, 10), width=2)
    flat = Image.new("RGB", (200, 300), MAGENTA)
    ImageDraw.Draw(flat).rectangle([50, 50, 150, 250], fill=(120, 60, 40))
    assert detail(reference) > detail(flat)
    assert is_flat(flat, reference)
    assert not is_flat(reference, reference)
