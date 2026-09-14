import numpy
import pytest
from PIL import Image

from cag.geometry import CELL_HEIGHT, CELL_WIDTH, CONTACT_ROW
from cag.mask import MaskError, key_art_scale, register, subject_box


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
