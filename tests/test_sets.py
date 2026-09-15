"""The one file that switches renders on and off."""

import json

import pytest

from cag import sets, static_sheet
from cag.prompts import KEY_VIEW, VIEWS

ALL_SETS = sorted(sets.SET_PLANS)


@pytest.fixture
def switches(tmp_path, monkeypatch):
    """Point the config at a file this test owns, and hand back a way to write it."""

    def write(payload):
        path = tmp_path / "enabled.json"
        path.write_text(json.dumps(payload))
        monkeypatch.setattr(sets, "SWITCHES", path)
        return path

    return write


def test_a_set_switched_off_is_not_rendered(switches):
    switches({"animations": {"entrance": False, "flinch": False}})
    left = sets.wanted(ALL_SETS, "animations")
    assert "entrance" not in left and "flinch" not in left
    assert "dance" in left and "ko" in left


def test_a_set_the_file_never_mentions_stays_on(switches):
    """A set added to the pipeline renders without having to be listed first."""
    switches({"animations": {"entrance": False}})
    assert "shimmy" in sets.wanted([*ALL_SETS, "shimmy"], "animations")


def test_no_file_means_everything_is_on(tmp_path, monkeypatch):
    monkeypatch.setattr(sets, "SWITCHES", tmp_path / "absent.json")
    assert sets.wanted(ALL_SETS, "animations") == ALL_SETS


def test_the_order_given_is_the_order_kept(switches):
    switches({"animations": {}})
    assert sets.wanted(["ko", "dance", "sing"], "animations") == ["ko", "dance", "sing"]


def test_a_view_switched_off_is_not_drawn(switches):
    switches({"views": {"back": False}})
    assert "back" not in static_sheet.projection_views()
    assert "profile" in static_sheet.projection_views()


def test_the_key_art_cannot_be_switched_off(switches):
    """Every other render quotes it, and the working scale is measured off it."""
    switches({"views": {view: False for view in VIEWS}})
    assert sets.wanted([KEY_VIEW], "views", sets.REQUIRED_VIEWS) == [KEY_VIEW]


def test_the_shipped_file_only_names_things_that_exist():
    """A typo in the config would silently switch off nothing at all."""
    loaded = json.loads(sets.SWITCHES.read_text())
    assert set(loaded["animations"]) <= set(sets.SET_PLANS)
    assert set(loaded["views"]) <= set(VIEWS)
