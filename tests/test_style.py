import pytest

from cag.style import (
    BACKDROP,
    DEFAULT_DETAIL_LEVEL,
    DETAIL_LEVELS,
    SIDE_LANGUAGE,
    STYLE,
    detail_clause,
    detail_frame,
)


def test_the_style_is_the_arcade_contract_not_a_cartoon():
    assert "Capcom Street Fighter 2 style arcade pixel art" in STYLE
    assert "two or three discrete tones" in STYLE.lower()
    for softener in ("cel-shaded", "cartoon", "smooth", "gradient shading"):
        assert softener not in STYLE.lower().replace("no antialiasing, gradients", "")


def test_the_ten_step_scale_is_complete():
    assert sorted(DETAIL_LEVELS) == list(range(1, 11))
    assert DEFAULT_DETAIL_LEVEL in DETAIL_LEVELS


def test_detail_clause_names_the_level_and_its_description():
    clause = detail_clause(4)
    assert clause.startswith("Render at detail level 4.")
    assert "two-to-three-tone shading" in clause


def test_detail_clause_rejects_an_off_scale_level():
    with pytest.raises(KeyError, match="1-10"):
        detail_clause(0)


def test_only_the_level_with_a_shipped_frame_returns_one():
    assert detail_frame(4).name == "detail-level-04.png"
    assert detail_frame(4).exists()
    assert detail_frame(7) is None


def test_side_language_is_the_clause_the_old_profile_required():
    assert SIDE_LANGUAGE.startswith("Mandatory character-side convention:")
    assert "Never use stage-left or stage-right" in SIDE_LANGUAGE


def test_the_backdrop_clause_names_the_failure_modes_that_actually_happen():
    """Paraphrasing this lost four renders to a black stage with a vignette."""
    for named in ("scene", "vignette", "cast shadow", "checkerboard", "rounded corners"):
        assert named in BACKDROP, named
    assert "100% of all non-character space" in BACKDROP
    assert "gaps between hair, limbs, and props" in BACKDROP
    assert "#FF00FF" in BACKDROP


def test_every_generation_prompt_carries_the_backdrop_clause():
    from cag.prompts import FRAME_VIEWS, frame_prompt, view_prompt
    from cag.spec import load_spec

    spec = load_spec("specs/belter.json")
    assert BACKDROP in view_prompt(spec, "B", "front")
    assert BACKDROP in frame_prompt(spec, "B", "N", FRAME_VIEWS["front"], "a cue", "key")


def test_every_frame_prompt_defends_the_prop_against_the_pose_cue():
    """The cue names arms and the skeleton draws bare joints; the mic vanished."""
    from cag.prompts import FRAME_VIEWS, frame_prompt
    from cag.spec import load_spec

    prompt = frame_prompt(
        load_spec("specs/belter.json"),
        "B",
        "N",
        FRAME_VIEWS["front"],
        "character-right arm at chest height, elbow bent ~90 degrees.",
        "key",
        pose_reference=True,
    )
    assert "still in that same hand" in prompt
    assert "bare fist" in prompt
