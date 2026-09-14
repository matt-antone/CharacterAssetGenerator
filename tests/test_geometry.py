from cag.geometry import PX_PER_FOOT, parse_height, subject_height_px


def test_cell_height_is_seven_feet():
    assert PX_PER_FOOT == 80.0


def test_parse_height():
    assert parse_height("5' 9\"") == 69.0
    assert parse_height("6'") == 72.0


def test_subject_height_px():
    # 7'0" fills the cell; 5'9" is proportionally shorter.
    assert subject_height_px(84) == 560
    assert subject_height_px(69) == 460
