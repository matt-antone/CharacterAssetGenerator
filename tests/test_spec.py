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
    spec = load_spec("specs/velvet-lou.json")
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
    assert load_spec("specs/velvet-lou.json").detail_level == 4
    assert load_spec(write(tmp_path, detail_level=7)).detail_level == 7
    for bad in (0, 11, "four", 4.5):
        with pytest.raises(SpecError, match="detail_level must be"):
            load_spec(write(tmp_path, detail_level=bad))
