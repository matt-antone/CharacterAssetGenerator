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
    assert "32-bit arcade sprite art" in STYLE
    assert "four to six discrete banded tones" in STYLE
    for softener in ("cel-shaded", "cartoon"):
        assert softener not in STYLE.lower()


def test_both_eras_keep_what_the_masker_and_the_style_depend_on():
    """Palette depth is the only difference; the rest is load-bearing."""
    from cag.style import SIXTEEN_BIT

    for look in (STYLE, SIXTEEN_BIT):
        assert "visible pixel grid" in look
        assert "black silhouette outline about two pixels wide" in look
        assert "no soft glow" in look or "or soft glow" in look
        assert "Full body, head to feet, nothing cropped." in look


def test_the_ten_step_scale_is_complete():
    assert sorted(DETAIL_LEVELS) == list(range(1, 11))
    assert DEFAULT_DETAIL_LEVEL in DETAIL_LEVELS


def test_detail_clause_names_the_level_and_its_description():
    clause = detail_clause(DEFAULT_DETAIL_LEVEL)
    assert clause.startswith(f"Render at detail level {DEFAULT_DETAIL_LEVEL}.")
    assert DETAIL_LEVELS[DEFAULT_DETAIL_LEVEL] in clause


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

    spec = load_spec("specs/default/belter.json")
    assert BACKDROP in view_prompt(spec, "B", "front")
    assert BACKDROP in frame_prompt(spec, "B", "N", FRAME_VIEWS["front"], "a cue", "key")


def test_a_frame_that_holds_a_prop_defends_it_against_the_pose_cue():
    """The cue names arms and the skeleton draws bare joints; the mic vanished.
    A set with empty hands is not told to keep a prop: belter's dance grew a mic."""
    from cag.prompts import FRAME_VIEWS, frame_prompt
    from cag.spec import load_spec

    args = (
        load_spec("specs/default/belter.json"),
        "B",
        "N",
        FRAME_VIEWS["front"],
        "character-right arm at chest height, elbow bent ~90 degrees.",
        "key",
    )
    held = frame_prompt(*args, pose_reference=True, props="She holds a microphone.")
    assert "still in that same hand" in held and "bare fist" in held
    assert "still in that same hand" not in frame_prompt(*args, pose_reference=True)


def test_every_render_is_told_where_the_viewer_stands():
    """Outlaw's victory set looked up at her; every other set looked slightly down."""
    from cag.prompts import frame_prompt, frame_sheet_prompt, view_prompt
    from cag.style import VIEWPOINT

    assert VIEWPOINT in STYLE  # so it reaches every prompt that carries the style
    assert "never looking up at the figure from below" in VIEWPOINT
    # Stated as the viewer's position, so a KO frame on the floor cannot drag the camera down.
    assert "This is the camera, not the pose" in VIEWPOINT


def test_the_costume_anchor_defends_footwear_before_hair():
    """A bible that describes hair at length would otherwise fill the anchor with
    it and leave the feet undefended — and feet are the measured failure: a lifted
    foot came back in the reference dancer's white trainer, not the character's boot."""
    from cag.prompts import costume_anchor

    bible = (
        "She is tall and lean. Thick auburn hair falls loose and ungathered. "
        "Hair uses #593029 and highlight #8A4A3F. "
        "Chunky ankle boots have near-black brown leather and warm mid-brown cords."
    )
    anchor = costume_anchor(bible)
    assert "ankle boots" in anchor
    assert "auburn hair" in anchor
    assert "tall and lean" not in anchor  # only what a photograph can contradict
    assert "still wears the character's own footwear" in anchor
    assert costume_anchor("She is tall and lean. She looks cheerful.") == ""


def test_the_costume_anchor_carries_the_whole_outfit_from_the_brief():
    """Two bible sentences left Diva's gown out of her dance prompt, and against a
    dancer in trousers she stepped a leg through a slit the gown does not have."""
    from cag.prompts import costume_anchor

    bible = "Low-heeled satin shoes in violet. Her hair is a high sculpted updo."
    gown = "A floor-length one-shoulder gown falling in a straight column; legs stay hidden."
    anchor = costume_anchor(bible, outfit=gown, hair="A high sculpted black updo.")
    assert gown in anchor
    assert "satin shoes" in anchor, "an outfit naming no footwear keeps the bible's"
    assert "sculpted black updo" in anchor and "high sculpted updo" not in anchor
    assert "keeps covering them" in anchor

    shod = costume_anchor(bible, outfit="Jeans and chunky ankle boots.")
    assert "satin shoes" not in shod, "the outfit's own footwear is enough"


def test_the_key_art_quotes_the_brief_s_own_stance():
    """READY_STANCE says "as described above"; without this the key art drew
    arms hanging and seven halloween keys lost their signature gesture.
    Projection views keep the neutral stance so they stay comparable."""
    from cag.prompts import KEY_VIEW, view_prompt
    from cag.spec import load_spec

    spec = load_spec("specs/halloween/jack.json")
    assert spec.performance_style in view_prompt(spec, "B", KEY_VIEW)
    assert spec.performance_style not in view_prompt(spec, "B", "front")


def test_the_key_art_keeps_both_hands_and_the_whole_costume():
    """A brief's own stance used to switch off the readable-arms line, and
    Belter's free hand went behind her back; two rolls also dropped her waist
    accent. Both rules are the key art's alone: projection views keep their
    neutral stance, and a set's own pose is not overridden."""
    from cag.prompts import KEY_ARMS, KEY_COMPLETE, KEY_VIEW, view_prompt
    from cag.spec import load_spec

    spec = load_spec("specs/default/belter.json")
    assert spec.performance_style, "the case this guards is a brief with its own stance"
    key = view_prompt(spec, "B", KEY_VIEW)
    assert KEY_ARMS in key and KEY_COMPLETE in key
    front = view_prompt(spec, "B", "front")
    assert KEY_ARMS not in front and KEY_COMPLETE not in front
    assert KEY_ARMS not in view_prompt(spec, "B", KEY_VIEW, pose="arms crossed")
