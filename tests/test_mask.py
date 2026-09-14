import numpy
import pytest
from PIL import Image

from cag.geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW
from cag.mask import (
    MaskError,
    backdrop_colour,
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
