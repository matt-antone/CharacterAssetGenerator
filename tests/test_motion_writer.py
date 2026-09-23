import json
from dataclasses import replace

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from cag.motion import MotionError, load_motion
from cag.motion_writer import parse, request, write_motion
from cag.sets import SET_PLANS, plan_for
from cag.spec import load_spec

SPEC = load_spec("specs/default/belter.json")
PLAN = plan_for("victory")


def motion_sheet(frames=PLAN.frame_count):
    return json.dumps(
        {
            "arc": "She plants, lifts her chin, and holds.",
            "frames": [
                {
                    "i": i,
                    "role": "key" if i in (0, 4) else "inbetween",
                    "pace": "steady",
                    "cue": f"Weight centred over both feet. stance {i}. head forward.",
                }
                for i in range(frames)
            ],
        }
    )


def model(*replies):
    return FakeMessagesListChatModel(responses=[AIMessage(r) for r in replies])


def test_writes_a_sheet_the_loader_accepts(tmp_path):
    dst = tmp_path / "victory.json"
    written = write_motion(model(motion_sheet()), SPEC, "victory", "Plant and hold.", PLAN, dst)
    assert written.fps == PLAN.fps and written.view == PLAN.view
    assert len(written.frames) == PLAN.frame_count
    assert not written.loops  # victory is a one-shot
    assert load_motion(dst).arc.startswith("She plants")


def test_a_written_sheet_has_no_landmarks_so_no_skeleton_is_drawn(tmp_path):
    written = write_motion(model(motion_sheet()), SPEC, "victory", "Plant.", PLAN, tmp_path / "v.json")
    assert not written.has_poses


def test_an_existing_sheet_is_reused_rather_than_rewritten(tmp_path):
    dst = tmp_path / "victory.json"
    write_motion(model(motion_sheet()), SPEC, "victory", "Plant.", PLAN, dst)
    before = dst.read_text()
    reused = write_motion(model("nonsense"), SPEC, "victory", "Plant.", PLAN, dst)
    assert dst.read_text() == before and len(reused.frames) == PLAN.frame_count


def test_a_fenced_reply_is_still_read():
    assert parse("```json\n{\"frames\": []}\n```") == {"frames": []}
    assert parse("here you go {\"a\": 1} thanks") == {"a": 1}


def test_a_reply_with_no_json_is_refused():
    with pytest.raises(MotionError, match="no JSON object"):
        parse("I could not do that.")


def test_a_short_sheet_is_retried_then_refused(tmp_path):
    with pytest.raises(MotionError, match="could not write a motion sheet"):
        write_motion(
            model(motion_sheet(frames=3), motion_sheet(frames=2)), SPEC, "victory", "Plant.", PLAN,
            tmp_path / "v.json",
        )
    assert not (tmp_path / "v.json").exists()


def test_a_bad_first_reply_is_retried(tmp_path):
    written = write_motion(
        model("not json at all", motion_sheet()), SPEC, "victory", "Plant.", PLAN, tmp_path / "v.json"
    )
    assert len(written.frames) == PLAN.frame_count


def test_the_request_tells_a_loop_to_close_and_a_one_shot_to_stop():
    assert "loops" in request(SPEC, "dance", "A two-step.", plan_for("dance"))
    assert "plays once and stops" in request(SPEC, "victory", "Plant.", plan_for("victory"))
    assert "settled resting state" in request(SPEC, "ko", "Fold safely.", plan_for("ko"))


def test_a_pingpong_plan_asks_for_a_movement_with_two_ends(tmp_path):
    """A written set bounces on the same word a traced one does."""
    plan = replace(plan_for("dance"), playback="pingpong")
    assert "pingpongs" in request(SPEC, "dance", "A two-step.", plan)

    written = write_motion(model(motion_sheet()), SPEC, "dance", "A sway.", plan, tmp_path / "d.json")
    assert written.playback == "pingpong"
    assert not written.loops  # it turns around; it does not cut back to frame 0


def test_every_set_in_the_belter_brief_has_a_plan():
    assert set(SPEC.sets) <= set(SET_PLANS)
    assert all(p.frame_count > 0 and p.fps > 0 for p in SET_PLANS.values())
