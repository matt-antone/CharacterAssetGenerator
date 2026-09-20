import json

import pytest

from cag.props import PropError, clauses, load_prop


def test_reads_the_microphone():
    mic = load_prop("mic")
    assert mic.hold["hands"] == {
        "character-right": mic.hold["hands"]["character-right"]
    }, "the microphone is held in one named hand"
    assert "grille" in mic.draw["shape"]


def test_a_set_gets_its_own_carriage():
    """Which is how the same prop is carried differently in sing and in dance."""
    mic = load_prop("mic")
    assert "toward the mouth" in mic.clause("sing")
    assert "free to move" in mic.clause("dance")


def test_a_set_the_prop_says_nothing_about_rests():
    """Key art is not performing anything, so the prop rests."""
    mic = load_prop("mic")
    assert mic.hold["rest"].rstrip(".") in mic.clause(None)
    assert mic.hold["rest"].rstrip(".") in mic.clause("a-set-with-no-carriage")


def test_holding_nothing_says_nothing():
    """A character whose brief names no props here is drawn empty-handed."""
    assert clauses((), "dance") == ""


def test_a_prop_that_is_not_there_says_so(tmp_path):
    with pytest.raises(PropError, match="no 'trumpet' prop"):
        load_prop("trumpet", tmp_path)


def test_a_prop_is_held_in_the_characters_own_hands(tmp_path):
    """Screen sides mirror with the figure; the character's own sides do not."""
    (tmp_path / "bad.json").write_text(json.dumps({
        "name": "bad", "summary": "x", "draw": {"shape": "x"},
        "hold": {"hands": {"left": "x"}},
    }))
    with pytest.raises(PropError, match="never a screen side"):
        load_prop("bad", tmp_path)


def test_a_two_handed_prop_names_both_hands(tmp_path):
    (tmp_path / "guitar.json").write_text(json.dumps({
        "name": "guitar", "summary": "One electric guitar.",
        "draw": {"shape": "A solid body about as wide as the character's torso."},
        "hold": {"hands": {
            "character-left": "Fingers stopped on the neck.",
            "character-right": "Strums across the bridge.",
        }},
    }))
    clause = load_prop("guitar", tmp_path).clause("sing")
    assert "character-left" in clause and "character-right" in clause
