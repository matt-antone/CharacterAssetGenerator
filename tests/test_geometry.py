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


def test_an_animation_frame_spans_eight_and_a_half_feet():
    assert ANIM_CELL_HEIGHT_INCHES == 102
    assert anim_subject_height_px(ANIM_CELL_HEIGHT_INCHES) == CELL_HEIGHT
    # 8'6" of world is less than the static cell's 9', so the character is larger here.
    assert anim_subject_height_px(69) > subject_height_px(69)


def test_the_animation_floor_sits_six_inches_off_the_bottom_edge():
    assert CELL_HEIGHT - ANIM_CONTACT_ROW == round(6 * ANIM_PX_PER_INCH)


# --- detail levels draw their own cells -------------------------------------

def test_the_legacy_geometry_is_the_560px_cell_builds_drew_before_levels():
    from cag import geometry
    legacy = geometry.LEGACY
    assert (legacy.cell, legacy.contact_row, legacy.anim_contact_row) == (
        geometry.CELL_HEIGHT, geometry.CONTACT_ROW, geometry.ANIM_CONTACT_ROW)
    assert geometry.current() is legacy, "outside a build nothing changes"


def test_a_level_draws_at_the_ladders_density_and_palette():
    from cag import geometry
    level8 = geometry.at_level(8)
    assert (level8.cell, level8.colours, level8.level) == (1342, 161, 8)
    # Ellis, the ladder's sample, stands at the ladder's height for the level.
    assert level8.subject_height_px(geometry.LADDER_FIGURE_INCHES) == 870
    cells = [geometry.at_level(level).cell for level in range(1, 11)]
    assert cells == sorted(cells) and len(set(cells)) == 10


def test_using_scopes_the_geometry_and_restores_it():
    from cag import geometry
    with geometry.using(geometry.at_level(8)):
        assert geometry.current().cell == 1342
        assert geometry.subject_height_px(67) == 833
    assert geometry.current() is geometry.LEGACY


def test_a_cut_out_lands_in_the_levels_cell_on_its_contact_row():
    from PIL import Image
    from cag import geometry, mask
    figure = Image.new("RGBA", (40, 100), (0, 0, 0, 0))
    figure.paste((30, 30, 30, 255), (10, 0, 30, 100))
    with geometry.using(geometry.at_level(8)) as g:
        cell = mask.register(figure, 3.0)
    assert cell.size == (g.cell, g.cell)
    assert cell.getbbox()[3] == g.contact_row + 1


def test_the_editor_reads_a_packages_cell_off_its_manifest(tmp_path):
    import json
    from cag import geometry
    from cag.edit import package_geometry
    assert package_geometry(tmp_path) is geometry.LEGACY, "no manifest: the 560px cells"
    (tmp_path / "manifest.json").write_text(json.dumps({"cell": [1342, 1342]}))
    assert package_geometry(tmp_path).cell == 1342
