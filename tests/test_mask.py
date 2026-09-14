import numpy
import pytest
from PIL import Image

from cag.geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW
from cag.mask import (
    MaskError,
    backdrop_colour,
    key_out,
    key_art_scale,
    register,
    subject_box,
    unmix,
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


def test_unmix_clears_backdrop_the_subject_encloses():
    """The gap inside a hand holding a microphone comes back opaque."""
    image = Image.new("RGBA", (2, 1))
    image.putpixel((0, 0), (255, 0, 255, 255))    # trapped backdrop
    image.putpixel((1, 0), (216, 80, 160, 255))   # hot pink costume, kept
    out = unmix(image, MAGENTA)
    assert out.getpixel((0, 0)) == (0, 0, 0, 0)
    assert out.getpixel((1, 0))[3] == 255


def test_unmix_divides_the_backdrop_back_out_of_a_blended_edge():
    """A half-covered edge pixel must recover the subject's own colour."""
    subject = numpy.array([0.0, 0.0, 0.0])  # a black outline
    blended = 0.5 * subject + 0.5 * MAGENTA
    image = Image.new("RGBA", (1, 1))
    image.putpixel((0, 0), (*[int(v) for v in blended], 128))
    red, green, blue, alpha = unmix(image, MAGENTA).getpixel((0, 0))
    assert max(red, green, blue) <= 4  # the magenta share is gone
    assert alpha == 128


def test_unmix_works_the_same_against_a_black_backdrop():
    """Vision segments the subject, so the backdrop colour is incidental."""
    subject = numpy.array([200.0, 40.0, 40.0])
    blended = 0.5 * subject + 0.5 * BLACK
    image = Image.new("RGBA", (1, 1))
    image.putpixel((0, 0), (*[int(v) for v in blended], 128))
    red, green, blue, _ = unmix(image, BLACK).getpixel((0, 0))
    assert abs(red - 200) <= 2 and abs(green - 40) <= 2 and abs(blue - 40) <= 2


def test_unmix_rounds_a_near_solid_subject_up_to_opaque():
    image = Image.new("RGBA", (2, 1))
    image.putpixel((0, 0), (200, 40, 40, 254))
    image.putpixel((1, 0), (200, 40, 40, 120))
    out = unmix(image, BLACK)
    assert out.getpixel((0, 0))[3] == 255
    assert out.getpixel((1, 0))[3] == 120


def belter_ish(size=(60, 80), backdrop=(247, 4, 248)):
    """A sprite-like figure on a sampled-magenta backdrop."""
    image = Image.new("RGB", size, backdrop)
    image.paste((192, 23, 35), (20, 20, 40, 45))   # crimson jacket
    image.paste((75, 87, 111), (22, 45, 38, 70))   # denim
    for x in range(19, 41):                         # black outline along the top
        image.putpixel((x, 19), (0, 0, 0))
    return image


def test_key_out_removes_the_backdrop_and_keeps_the_black_outline():
    image = belter_ish()
    out = key_out(image, backdrop_colour(image))
    assert out.getpixel((0, 0)) == (0, 0, 0, 0)          # backdrop cleared
    assert out.getpixel((30, 30)) == (192, 23, 35, 255)  # jacket untouched
    assert out.getpixel((30, 19)) == (0, 0, 0, 255)      # outline survives


def test_key_out_pulls_a_faint_edge_back_towards_the_subject_colour():
    """The ramp estimates coverage rather than solving it, so this is a move in
    the right direction, not an exact recovery. It is what stops the fringe."""
    backdrop = numpy.array([247.0, 4.0, 248.0])
    jacket = numpy.array([192.0, 23.0, 35.0])
    blend = 0.25 * jacket + 0.75 * backdrop
    image = Image.new("RGB", (1, 1), tuple(int(v) for v in blend))
    red, green, blue, alpha = key_out(image, backdrop).getpixel((0, 0))

    before = numpy.max(numpy.abs(blend - jacket))
    after = numpy.max(numpy.abs(numpy.array([red, green, blue]) - jacket))
    assert 0 < alpha < 255
    assert after < before  # measured: about a fifth of the backdrop share comes off


def test_a_half_covered_high_contrast_edge_reads_as_solid():
    """The documented ceiling of a fixed ramp — see SUBJECT_DISTANCE."""
    backdrop = numpy.array([247.0, 4.0, 248.0])
    blend = 0.5 * numpy.array([192.0, 23.0, 35.0]) + 0.5 * backdrop
    image = Image.new("RGB", (1, 1), tuple(int(v) for v in blend))
    assert key_out(image, backdrop).getpixel((0, 0))[3] == 255


def test_key_out_leaves_far_fewer_soft_pixels_than_a_semantic_mask():
    """Pixel art has hard edges; a colour key preserves them."""
    out = numpy.array(key_out(belter_ish(), numpy.array([247.0, 4.0, 248.0])))
    alpha = out[:, :, 3]
    soft = int(((alpha > 0) & (alpha < 255)).sum())
    assert soft < int((alpha == 255).sum()) // 10


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
