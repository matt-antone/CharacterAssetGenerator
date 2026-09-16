from cag.geometry import (
    ANIM_CELL_HEIGHT_INCHES,
    ANIM_CONTACT_ROW,
    ANIM_PX_PER_INCH,
    CELL_HEIGHT,
    CELL_HEIGHT_INCHES,
    PX_PER_FOOT,
    anim_subject_height_px,
    parse_height,
    subject_height_px,
)


def test_cell_height_is_nine_feet():
    assert CELL_HEIGHT_INCHES == 108
    assert PX_PER_FOOT == CELL_HEIGHT / 9


def test_parse_height():
    assert parse_height("5' 9\"") == 69.0
    assert parse_height("6'") == 72.0


def test_subject_height_px():
    # The full cell height is 9'0"; anyone shorter is proportionally shorter.
    assert subject_height_px(CELL_HEIGHT_INCHES) == CELL_HEIGHT
    assert subject_height_px(69) == round(69 * CELL_HEIGHT / CELL_HEIGHT_INCHES)


def test_an_animation_frame_spans_a_foot_more_world_than_the_static_cell():
    assert ANIM_CELL_HEIGHT_INCHES == CELL_HEIGHT_INCHES + 12
    assert anim_subject_height_px(ANIM_CELL_HEIGHT_INCHES) == CELL_HEIGHT
    # The same character is smaller in an animation frame than on a static sheet.
    assert anim_subject_height_px(69) < subject_height_px(69)


def test_the_animation_floor_sits_six_inches_off_the_bottom_edge():
    assert CELL_HEIGHT - ANIM_CONTACT_ROW == round(6 * ANIM_PX_PER_INCH)
