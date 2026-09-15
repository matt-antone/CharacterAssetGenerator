from cag.geometry import (
    ANIM_CONTACT_ROW,
    ANIM_PX_PER_INCH,
    CELL_HEIGHT,
    PX_PER_FOOT,
    anim_subject_height_px,
    parse_height,
    subject_height_px,
)


def test_cell_height_is_seven_feet():
    assert PX_PER_FOOT == 80.0


def test_parse_height():
    assert parse_height("5' 9\"") == 69.0
    assert parse_height("6'") == 72.0


def test_subject_height_px():
    # 7'0" fills the cell; 5'9" is proportionally shorter.
    assert subject_height_px(84) == 560
    assert subject_height_px(69) == 460


def test_an_animation_frame_spans_eight_feet_not_seven():
    assert anim_subject_height_px(96) == CELL_HEIGHT
    # The same character is smaller in an animation frame than on a static sheet.
    assert anim_subject_height_px(69) < subject_height_px(69)


def test_the_animation_floor_sits_six_inches_off_the_bottom_edge():
    assert CELL_HEIGHT - ANIM_CONTACT_ROW == round(6 * ANIM_PX_PER_INCH)
