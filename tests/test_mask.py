import json
import statistics
from pathlib import Path
import numpy
import pytest
from PIL import Image

from cag.geometry import (
    ANIM_CONTACT_ROW,
    CELL_HEIGHT,
    CELL_WIDTH,
    CONTACT_ROW,
    anim_subject_height_px,
    subject_height_px,
)
from cag.mask import crown, pose_extent, stature
from cag.mask import (
    KEY_THRESHOLD,
    MaskError,
    backdrop_colour,
    backdrop_share,
    cut_in,
    frame_scale,
    key_out,
    key_art_scale,
    keyable,
    register,
    set_to_cells,
    subject_box,
    trim_matte,
)


def figure(size=(100, 200), box=(40, 20, 60, 180)):
    """An RGBA image with one opaque rectangle standing in for a character."""
    image = Image.new("RGBA", size, (0, 0, 0, 0))
    image.paste((10, 10, 10, 255), box)
    return image


def test_subject_box_ignores_near_transparent_pixels():
    image = figure()
    image.putpixel((0, 0), (255, 0, 0, 4))
    assert subject_box(image) == (40, 20, 60, 180)


def test_subject_box_rejects_empty_cutout():
    with pytest.raises(MaskError, match="fully transparent"):
        subject_box(Image.new("RGBA", (10, 10), (0, 0, 0, 0)))


def test_key_art_scale_maps_subject_to_physical_height():
    # 160px of drawn subject must become 5'9" at the cell's own scale.
    assert key_art_scale(figure(), 69) == pytest.approx(subject_height_px(69) / 160)


def test_register_puts_feet_on_the_contact_row_and_centres():
    cell = register(figure(), key_art_scale(figure(), 69))
    assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
    left, top, right, bottom = subject_box(cell)
    # The bbox is exclusive, so the last visible row is the contact row itself.
    assert bottom - 1 == CONTACT_ROW
    assert bottom - top == subject_height_px(69)
    assert (left + right) // 2 == pytest.approx(CELL_WIDTH // 2, abs=1)


def test_register_keeps_one_scale_across_poses():
    """A crouch must render shorter, not be stretched back to full height."""
    scale = key_art_scale(figure(), 69)
    crouch = register(figure(box=(40, 100, 60, 180)), scale)
    left, top, right, bottom = subject_box(crouch)
    assert bottom - 1 == CONTACT_ROW
    assert bottom - top == subject_height_px(69) // 2  # half the standing span, exactly


def test_register_does_not_crash_when_subject_overflows_the_cell():
    cell = register(figure(), key_art_scale(figure(), 84))
    assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
    assert subject_box(cell)[3] - 1 == CONTACT_ROW


def test_register_preserves_soft_edges():
    """Pasting must not multiply alpha by itself and eat antialiased pixels."""
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    image.paste((10, 10, 10, 255), (5, 5, 15, 14))
    for x in range(5, 15):
        image.putpixel((x, 14), (10, 10, 10, 40))  # a soft bottom edge
    alpha = numpy.array(register(image, 1.0).getchannel("A"))
    assert alpha.max() == 255
    soft = alpha[(alpha > 0) & (alpha < 255)]
    assert soft.size and 30 <= soft.min() <= 50  # not squared down to ~6


MAGENTA = numpy.array([255.0, 0.0, 255.0])
BLACK = numpy.array([0.0, 0.0, 0.0])


def test_backdrop_colour_is_measured_from_the_border_not_assumed():
    image = Image.new("RGB", (40, 40), (12, 10, 14))
    image.paste((200, 40, 40), (10, 10, 30, 30))  # a subject in the middle
    assert list(backdrop_colour(image)) == [12, 10, 14]


def test_trim_matte_clears_backdrop_the_subject_encloses():
    """The gap inside a hand holding a microphone comes back transparent."""
    image = Image.new("RGBA", (9, 9), (216, 80, 160, 255))  # hot pink costume
    image.putpixel((4, 4), (255, 0, 255, 255))              # trapped backdrop
    out = trim_matte(image, MAGENTA)
    assert out.getpixel((4, 4)) == (0, 0, 0, 0)


def test_a_costume_in_the_backdrop_hue_survives():
    """A hot pink character must not come out translucent or eaten.

    It measures 0.31 against the backdrop, well under the threshold, so the
    in-or-out decision keeps it whole. This is the reason the Vision path and
    the channel-spread measure exist at all.
    """
    image = Image.new("RGBA", (9, 9), (216, 80, 160, 255))
    assert trim_matte(image, MAGENTA).getpixel((4, 4)) == (216, 80, 160, 255)


def test_trim_matte_leaves_vision_in_charge_of_an_achromatic_backdrop():
    """Nothing to key against, so Vision's own matte stands, cut in by a pixel."""
    image = Image.new("RGBA", (9, 9), (200, 40, 40, 255))
    out = trim_matte(image, BLACK)
    assert out.getpixel((4, 4)) == (200, 40, 40, 255)  # interior untouched
    assert out.getpixel((0, 0)) == (0, 0, 0, 0)        # a pixel off the edge


def test_cut_in_takes_one_pixel_off_every_edge():
    keep = numpy.zeros((7, 7), dtype=bool)
    keep[2:5, 2:5] = True        # a 3x3 block
    assert int(cut_in(keep).sum()) == 1  # cut back to its centre


def test_the_blended_rim_survives_the_threshold_and_so_must_be_cut():
    """Why CUT_IN exists rather than a cleverer threshold.

    A rim pixel half backdrop and half black outline measures just under the
    threshold -- indistinguishable from genuinely dark costume in the backdrop's
    hue. No threshold separates them, so the rim is discarded by position.
    """
    rim = numpy.array([[[123.0, 2.0, 124.0]]])  # half (247, 4, 248), half black
    assert backdrop_share(rim, numpy.array([247.0, 4.0, 248.0]))[0, 0] < KEY_THRESHOLD


def belter_ish(size=(60, 80), backdrop=(247, 4, 248)):
    """A sprite-like figure on a sampled-magenta backdrop.

    Drawn with the two-pixel outline STYLE mandates, plus the half-magenta rim a
    real render leaves where that outline was antialiased against the backdrop.
    """
    image = Image.new("RGB", size, backdrop)
    image.paste((123, 2, 124), (19, 19, 41, 71))  # blended rim, half backdrop
    image.paste((0, 0, 0), (20, 20, 40, 70))      # two-pixel black outline
    image.paste((192, 23, 35), (22, 22, 38, 45))  # crimson jacket
    image.paste((75, 87, 111), (22, 45, 38, 68))  # denim
    return image


def test_key_out_clears_the_backdrop_and_keeps_the_outline():
    out = key_out(belter_ish(), backdrop_colour(belter_ish()))
    assert out.getpixel((0, 0)) == (0, 0, 0, 0)          # backdrop cleared
    assert out.getpixel((30, 19)) == (0, 0, 0, 0)        # blended rim cut away
    assert out.getpixel((30, 20)) == (0, 0, 0, 255)      # outline survives it
    assert out.getpixel((30, 30)) == (192, 23, 35, 255)  # crimson jacket kept


def test_key_out_leaves_no_backdrop_tint_along_the_outline():
    """The reported bug: magenta survived all the way round the character."""
    out = numpy.array(key_out(belter_ish(), backdrop_colour(belter_ish())))
    rgb, alpha = out[:, :, :3], out[:, :, 3]
    visible = alpha > 0
    red, green, blue = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
    magenta = visible & (red > 100) & (blue > 100) & (numpy.minimum(red, blue) - green > 45)
    assert int(magenta.sum()) == 0


def test_key_out_keys_nothing_against_an_achromatic_backdrop():
    """Black, white and grey give the key no channel spread to work with.

    It must key nothing at all, so `cutout` reaches Vision -- rather than find
    no backdrop and keep the entire canvas as character.
    """
    image = Image.new("RGB", (20, 20), (0, 0, 0))
    image.paste((200, 40, 40), (5, 5, 15, 15))
    assert not keyable(numpy.array([0.0, 0.0, 0.0]))
    assert key_out(image, numpy.array([0.0, 0.0, 0.0])).getbbox() is None


def test_key_out_produces_no_soft_pixels_at_all():
    """Pixel art has hard edges, and `register` resamples nearest-neighbour, so
    the matte is a straight in-or-out decision with nothing in between."""
    alpha = numpy.array(key_out(belter_ish(), numpy.array([247.0, 4.0, 248.0])))[:, :, 3]
    assert int(((alpha > 0) & (alpha < 255)).sum()) == 0


def test_cutout_falls_back_to_vision_when_the_key_finds_nothing(tmp_path, monkeypatch):
    """An all-backdrop render keys to nothing, so the semantic mask takes over."""
    from cag import mask as mask_module

    blank = tmp_path / "blank.png"
    Image.new("RGB", (40, 40), (247, 4, 248)).save(blank)
    sentinel = Image.new("RGBA", (40, 40), (1, 2, 3, 255))
    monkeypatch.setattr(mask_module, "vision_cutout", lambda src: sentinel)
    assert mask_module.cutout(blank) is sentinel


def test_cutout_uses_the_key_when_it_finds_a_subject(tmp_path, monkeypatch):
    from cag import mask as mask_module

    path = tmp_path / "figure.png"
    belter_ish().save(path)
    monkeypatch.setattr(
        mask_module, "vision_cutout", lambda src: pytest.fail("should not need Vision")
    )
    assert numpy.array(mask_module.cutout(path))[:, :, 3].max() == 255


#: A standing skeleton, enough of one for a scale to be read off it.
POSE = {
    "anL": [48, 180], "anR": [52, 180],
    "knL": [48, 140], "knR": [52, 140],
    "hipL": [48, 100], "hipR": [52, 100],
    "shL": [46, 60], "shR": [54, 60],
    "elL": [44, 85], "elR": [56, 85],
    "wrL": [43, 105], "wrR": [57, 105],
    "earL": [48, 45], "earR": [52, 45], "nose": [50, 47],
}


def test_frame_scale_ignores_how_big_the_generator_drew_the_frame():
    """The animation bug itself: every frame is a separate generation, drawn at
    whatever size, and a fixed factor piped that size straight into the cell."""
    small = figure(size=(200, 400), box=(80, 40, 120, 360))
    large = figure(size=(400, 800), box=(160, 80, 240, 720))  # same pose, twice over
    heights = []
    for art in (small, large):
        cell = register(art, frame_scale(art, POSE, 182, 140, 69), ANIM_CONTACT_ROW)
        top, bottom = subject_box(cell)[1], subject_box(cell)[3]
        heights.append(bottom - top)
    assert heights[0] == pytest.approx(heights[1], abs=1)


def test_frame_scale_puts_a_character_at_their_height_in_the_eight_foot_frame():
    art = figure(size=(200, 400), box=(80, 40, 120, 360))
    cell = register(art, frame_scale(art, POSE, 182, 140, 69), ANIM_CONTACT_ROW)
    top, bottom = subject_box(cell)[1], subject_box(cell)[3]
    # The drawn box spans the whole pose, of which stature is the part that is
    # height; scaled back up it must be 5'9" on the animation frame's ruler.
    implied = (bottom - top) * stature(POSE, 140) / pose_extent(POSE, 182, 140)
    assert implied == pytest.approx(anim_subject_height_px(69), abs=2)


def test_animation_frames_stand_on_the_animation_contact_row():
    cell = register(figure(), key_art_scale(figure(), 69), ANIM_CONTACT_ROW)
    assert subject_box(cell)[3] - 1 == ANIM_CONTACT_ROW
    assert ANIM_CONTACT_ROW < CONTACT_ROW  # further up: the frame keeps a margin


def figure_sheet(path, boxes, size=(400, 300)):
    """A magenta sheet with one flat grey figure per box, shade rising in list order."""
    sheet = Image.new("RGB", size, (255, 0, 255))
    for n, box in enumerate(boxes):
        sheet.paste((30 + 20 * n,) * 3, box)
    sheet.save(path)
    return path


def test_slice_sheet_finds_figures_in_reading_order(tmp_path):
    from cag.mask import slice_frame_sheet

    # Two rows of two, the second row shifted so a fixed grid would miss it.
    boxes = [(20, 20, 80, 130), (150, 30, 220, 130), (40, 170, 90, 280), (250, 160, 330, 280)]
    cells = slice_frame_sheet(figure_sheet(tmp_path / "sheet.png", boxes), 4)
    shades = [cell.getpixel((cell.width // 2, cell.height // 2))[0] for cell in cells]
    assert shades == [30, 50, 70, 90]
    for cell, (l, t, r, b) in zip(cells, boxes):
        assert (cell.width, cell.height) == (r - l + 32, b - t + 32)
        assert cell.getpixel((0, 0)) == (255, 0, 255)


def test_slice_sheet_bridges_a_gap_inside_one_figure(tmp_path):
    from cag.mask import slice_frame_sheet

    # A raised arm 6px clear of the torso is still one figure.
    boxes = [(20, 40, 60, 130), (66, 20, 76, 80)]
    cells = slice_frame_sheet(figure_sheet(tmp_path / "sheet.png", boxes), 1)
    assert cells[0].width == 76 - 20 + 32


def test_slice_sheet_rejects_the_wrong_figure_count(tmp_path):
    from cag.mask import slice_frame_sheet

    with pytest.raises(MaskError, match="holds 2 figures, not 3"):
        slice_frame_sheet(figure_sheet(tmp_path / "sheet.png", [(20, 20, 60, 100), (100, 20, 140, 100)]), 3)


from cag.mask import slice_frame_sheet  # noqa: E402


def sheet_with_overlapping_rows(path, backdrop=(247, 4, 248)):
    """Eight figures in two rows of four, the way a sheet render lays them out.

    The first figure of the second row raises a hand up beside the boots of the
    figure above it, so no row of backdrop runs clear across the sheet between
    the two rows. Splitting by rows and columns reads that column as one figure.
    """
    image = Image.new("RGB", (400, 400), backdrop)
    for row in range(2):
        for column in range(4):
            x, y = 10 + column * 100, 10 + row * 200
            image.paste((0, 0, 0), (x, y, x + 60, y + 180))
    # The hand clears the figure above it by 2px, so the two never touch, but it
    # reaches past that figure's feet and leaves no clear row between the rows.
    image.paste((0, 0, 0), (72, 170, 92, 215))
    image.save(path)
    return path


def test_a_raised_hand_beside_the_row_above_still_counts_eight_figures(tmp_path):
    sheet = sheet_with_overlapping_rows(tmp_path / "sheet-00.png")
    assert len(slice_frame_sheet(sheet, 8)) == 8


def test_a_detached_piece_counts_with_its_figure_not_as_one(tmp_path):
    image = Image.new("RGB", (400, 400), (247, 4, 248))
    for column in range(4):
        x = 10 + column * 100
        image.paste((0, 0, 0), (x, 10, x + 60, 190))
        image.paste((0, 0, 0), (x + 20, 200, x + 80, 380))
    image.paste((0, 0, 0), (14, 2, 22, 8))  # a hair spike clear of the head
    sheet = image.save(p := tmp_path / "sheet-00.png") or p
    assert len(slice_frame_sheet(sheet, 8)) == 8


def test_a_sheet_drawn_short_is_still_rejected(tmp_path):
    image = Image.new("RGB", (400, 400), (247, 4, 248))
    for column in range(4):  # four figures where eight were asked for
        x = 10 + column * 100
        image.paste((0, 0, 0), (x, 10, x + 60, 190))
    sheet = image.save(p := tmp_path / "sheet-00.png") or p
    with pytest.raises(MaskError, match="holds 4 figures, not 8"):
        slice_frame_sheet(sheet, 8)


def test_raised_arms_do_not_shrink_a_sheet_with_no_landmarks(tmp_path):
    """Frank's victory: half the sheet reached overhead, the median box grew with
    the arms, and every frame of the set came out small."""
    sources = {}
    for index in range(8):
        top = 20 if index in (3, 4, 5, 6) else 50  # four frames reach 30px above the head
        image = Image.new("RGB", (100, 300), (247, 4, 248))
        image.paste((0, 0, 0), (40, top, 60, 250))
        image.save(sources.setdefault(index, tmp_path / f"{index:02d}.png"))
    cells = set_to_cells(sources, lambda i: tmp_path / "cells" / f"{i:02d}.png", {}, 0, 0, 69)
    top, bottom = subject_box(Image.open(cells[0]))[1], subject_box(Image.open(cells[0]))[3]
    assert abs((bottom - top) - anim_subject_height_px(69)) <= 2


def test_one_crouched_frame_does_not_grow_a_sheet_with_no_landmarks(tmp_path):
    """A guard that drops low on one frame of eight must not become the ruler
    for the seven standing ones."""
    sources = {}
    for index in range(8):
        top = 110 if index == 4 else 50  # one frame crouches 60px below the head
        image = Image.new("RGB", (100, 300), (247, 4, 248))
        image.paste((0, 0, 0), (40, top, 60, 250))
        image.save(sources.setdefault(index, tmp_path / f"{index:02d}.png"))
    cells = set_to_cells(sources, lambda i: tmp_path / "cells" / f"{i:02d}.png", {}, 0, 0, 69)
    top, bottom = subject_box(Image.open(cells[0]))[1], subject_box(Image.open(cells[0]))[3]
    assert abs((bottom - top) - anim_subject_height_px(69)) <= 2


# --- pose measurement, moved here with `stature` and `pose_extent` -----------
#: A stand and a crouch built so every segment keeps its length between the
#: two: 40px shin, 40px thigh, 40px torso.
BODY_H = 140
FLOOR_Y = 182
STAND = {
    "anL": [48, 180], "anR": [52, 180],
    "knL": [48, 140], "knR": [52, 140],
    "hipL": [48, 100], "hipR": [52, 100],
    "shL": [46, 60], "shR": [54, 60],
    "elL": [44, 85], "elR": [56, 85],
    "wrL": [43, 105], "wrR": [57, 105],
    "earL": [48, 45], "earR": [52, 45], "nose": [50, 47],
}
CROUCH = {
    "anL": [48, 180], "anR": [52, 180],
    "knL": [72, 148], "knR": [76, 148],
    "hipL": [48, 116], "hipR": [52, 116],
    "shL": [46, 76], "shR": [54, 76],
    "elL": [44, 101], "elR": [56, 101],
    "wrL": [43, 121], "wrR": [57, 121],
    "earL": [48, 61], "earR": [52, 61], "nose": [50, 63],
}


def test_bending_a_knee_folds_the_leg_without_shortening_it():
    """The whole point of measuring along bones instead of down a bounding box."""
    assert stature(CROUCH, BODY_H) == pytest.approx(stature(STAND, BODY_H))


def test_the_same_crouch_does_shrink_the_vertical_extent():
    """What a bounding box would have measured, and why it misleads."""
    assert pose_extent(CROUCH, FLOOR_Y, BODY_H) < pose_extent(STAND, FLOOR_Y, BODY_H)


def test_the_crown_leans_with_the_head_rather_than_staying_overhead():
    tilted = dict(STAND, earL=[63, 52], earR=[67, 52], nose=[69, 54])
    upright, leaning = crown(STAND, BODY_H), crown(tilted, BODY_H)
    assert upright[0] == pytest.approx(50)  # squarely above the neck
    assert leaning[0] > upright[0] + 5  # carried sideways by the tilt
    assert leaning[1] > upright[1]  # and, being off the vertical, lower


def test_stature_tracks_extent_across_a_real_traced_set():
    """The landmarks are 2D, so a limb angled at the camera foreshortens and
    stature alone wobbles by ~9%. `frame_scale` divides it by the extent, and
    that ratio is several times steadier, because both shrink together."""
    motion = json.loads(Path("tests/fixtures/sample-motion.json").read_text())
    poses = [frame["pts"] for frame in motion["frames"]]
    statures = [stature(pose, motion["body_h"]) for pose in poses]
    ratios = [
        stature(pose, motion["body_h"]) / pose_extent(pose, motion["floor_y"], motion["body_h"])
        for pose in poses
    ]
    spread = lambda xs: (max(xs) - min(xs)) / statistics.median(xs)
    assert spread(statures) > 0.05  # the raw measure really does move about
    assert spread(ratios) < 0.05  # the one the scale is built on does not


def test_a_kicked_leg_does_not_shove_the_body_and_a_hop_leaves_the_floor(tmp_path):
    """Sheet frames were centred by their box and pinned by their lowest pixel: a
    leg out to one side pushed the torso the other way, and no jump survived."""
    kick = {**POSE, "anL": [90, 180], "knL": [70, 140]}
    hop = {name: [x, y - 14] for name, (x, y) in POSE.items()}  # 14 of 140: a tenth of body height
    sources = {}
    for index, leg in enumerate([None, (60, 170, 100, 180), None]):
        image = Image.new("RGB", (200, 300), (247, 4, 248))
        image.paste((0, 0, 0), (40, 20, 60, 180))
        if leg:
            image.paste((0, 0, 0), leg)
        image.save(sources.setdefault(index, tmp_path / f"{index:02d}.png"))
    cells = set_to_cells(
        sources, lambda i: tmp_path / "cells" / f"{i:02d}.png",
        {0: POSE, 1: kick, 2: hop}, 182, 140, 69, airborne=frozenset({2}),
    )
    boxes = [subject_box(Image.open(cells[index])) for index in range(3)]
    assert abs(boxes[1][0] - boxes[0][0]) <= 1  # torso's edge stays put; only the leg reaches out
    assert boxes[0][3] - 1 == ANIM_CONTACT_ROW
    lift = round((182 - 166) / 140 * anim_subject_height_px(69))
    assert boxes[0][3] - boxes[2][3] == pytest.approx(lift, abs=1)
