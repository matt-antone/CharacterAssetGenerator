import json

import pytest

from cag.spec import SpecError, load_spec

VALID = {
    "name": "Velvet Lou",
    "height": "5' 9\"",
    "description": "A lounge performer.",
    "animations": {"dance": "A two-step loop."},
}


def write(tmp_path, **overrides):
    data = {**VALID, **overrides}
    for key, value in list(data.items()):
        if value is None:
            del data[key]
    path = tmp_path / "spec.json"
    path.write_text(json.dumps(data))
    return path


def test_loads_the_sample_spec():
    spec = load_spec("tests/fixtures/velvet-lou.json")
    assert spec.slug == "velvet-lou"
    assert spec.height_inches == 69.0
    assert "dance" in spec.animations


def test_animations_are_optional(tmp_path):
    assert load_spec(write(tmp_path, animations=None)).animations == {}


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"name": "  "}, "non-empty name"),
        ({"height": "tall"}, "height must read"),
        ({"description": None}, "non-empty description"),
        ({"animations": {"dance": ""}}, "non-empty description"),
        ({"fps": 4}, "unknown fields: fps"),
        ({"name": "!!!"}, "no usable characters"),
    ],
)
def test_rejects_bad_briefs(tmp_path, overrides, message):
    with pytest.raises(SpecError, match=message):
        load_spec(write(tmp_path, **overrides))


def test_rejects_broken_json(tmp_path):
    path = tmp_path / "spec.json"
    path.write_text("{nope")
    with pytest.raises(SpecError, match="not valid JSON"):
        load_spec(path)


def test_detail_level_defaults_and_validates(tmp_path):
    from cag.style import DEFAULT_DETAIL_LEVEL

    assert load_spec("tests/fixtures/velvet-lou.json").detail_level == DEFAULT_DETAIL_LEVEL
    assert load_spec(write(tmp_path, detail_level=7)).detail_level == 7
    for bad in (0, 11, "four", 4.5):
        with pytest.raises(SpecError, match="detail_level must be"):
            load_spec(write(tmp_path, detail_level=bad))


def test_an_animation_may_name_the_sheet_that_drives_it(tmp_path):
    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Belter", "height": "5' 7\"", "description": "A rock singer.",
        "animations": {"dance": {"intent": "A club loop.", "motion": "zs-loop"},
                       "sing": "A held note."},
    }))
    spec = load_spec(brief)
    assert spec.animations == {"dance": "A club loop.", "sing": "A held note."}
    assert spec.motions == {"dance": "zs-loop"}


def test_a_brief_names_a_sheet_and_never_points_at_a_file(tmp_path):
    """A path in a brief only builds on the machine it was written on."""
    for bad in ("../MotionArtist/work/zs-loop", "/abs/zs-loop", "./zs-loop"):
        brief = tmp_path / "c.json"
        brief.write_text(json.dumps({
            "name": "Belter", "height": "5' 7\"", "description": "A rock singer.",
            "animations": {"dance": {"intent": "A club loop.", "motion": bad}},
        }))
        with pytest.raises(SpecError, match="not a path"):
            load_spec(brief)


def test_props_are_named_for_the_key_art_and_per_animation(tmp_path):
    """A prop named for sing and not for dance is how it is kept out of dance."""
    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Belter", "height": "5' 7\"", "description": "A rock singer.",
        "props": ["mic"],
        "animations": {"sing": {"intent": "A held note.", "props": ["mic"]},
                       "dance": "A club loop."},
    }))
    spec = load_spec(brief)
    assert spec.props == ("mic",)
    assert spec.animation_props == {"sing": ("mic",)}
    assert spec.animation_props.get("dance", ()) == (), "empty hands in dance"


def test_a_brief_names_props_and_never_points_at_files(tmp_path):
    brief = tmp_path / "c.json"
    brief.write_text(json.dumps({
        "name": "Belter", "height": "5' 7\"", "description": "A rock singer.",
        "props": ["../props/mic"],
        "animations": {"dance": "A club loop."},
    }))
    with pytest.raises(SpecError, match="not a prop"):
        load_spec(brief)
