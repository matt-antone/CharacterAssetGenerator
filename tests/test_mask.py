import numpy
import pytest
from PIL import Image

from cag.geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW
from cag.mask import (
    KEY_THRESHOLD,
    MaskError,
    backdrop_colour,
    backdrop_share,
    cut_in,
    key_out,
    key_art_scale,
    keyable,
    register,
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
    # 160px of subject must become 5'9" == 460px in the cell.
    assert key_art_scale(figure(), 69) == pytest.approx(460 / 160)


def test_register_puts_feet_on_the_contact_row_and_centres():
    cell = register(figure(), key_art_scale(figure(), 69))
    assert cell.size == (CELL_WIDTH, CELL_HEIGHT)
    left, top, right, bottom = subject_box(cell)
    # The bbox is exclusive, so the last visible row is the contact row itself.
    assert bottom - 1 == CONTACT_ROW
    assert bottom - top == 460
    assert (left + right) // 2 == pytest.approx(CELL_WIDTH // 2, abs=1)


def test_register_keeps_one_scale_across_poses():
    """A crouch must render shorter, not be stretched back to full height."""
    scale = key_art_scale(figure(), 69)
    crouch = register(figure(box=(40, 100, 60, 180)), scale)
    left, top, right, bottom = subject_box(crouch)
    assert bottom - 1 == CONTACT_ROW
    assert bottom - top == 230  # half the standing span, exactly


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
