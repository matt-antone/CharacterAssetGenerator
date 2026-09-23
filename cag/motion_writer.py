"""Write a motion sheet from prose when no video traced one.

MotionArtist produces sheets from real footage, with landmarks the keyframer can
be shown. Only some sets have footage. For the rest the brief's prose intent is
all there is, so a motion director writes the frame plan from it in the same
shape — which means the animation graph does not care where a sheet came from.

A written sheet carries no landmarks, so those sets get no pose skeleton. That
is a real quality difference, not a formality: the skeleton is the strongest
pose lever in the pipeline.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage

from .motion import MotionError, MotionSheet, load_motion
from .sets import SetPlan
from .spec import CharacterSpec

SYSTEM = """You are the motion director for one animation set of a game character. Write the \
frame-by-frame plan.

Return one JSON object and nothing else, in this exact shape:

{"arc": "<2-4 sentences on how the whole set reads>",
 "frames": [{"i": 0, "role": "key", "pace": "fast", "cue": "<the pose>"}, ...]}

Rules for the frames:
- Exactly the requested number, numbered from 0, in order.
- "role" is "key" for a hold or an extreme, "pilot" for the single fastest transition, and \
"inbetween" for the rest. Mark 2 to 4 keys, at most one pilot, and make frame 0 a key.
- "pace" is "fast", "steady" or "slow".
- "cue" describes one pose in plain declarative clauses, in this order: where the weight is, the \
stance, the legs, each arm, the torso, the hips or shoulders if they lift, and the head. Say what \
the body is doing, never what the character feels.
- Name the character's own sides as "character-left" and "character-right". Never "left", \
"right", "screen-left", "screen-right", "stage-left" or "stage-right".
- Describe the body only. Never mention costume, identity, props, background or camera.
- Distinct poses. Two adjacent frames must not read the same.

Output the JSON and nothing else."""


def request(spec: CharacterSpec, set_name: str, intent: str, plan: SetPlan) -> str:
    ending = {
        "loop": "The set loops: the last frame must lead naturally back into frame 0 as one "
        "more step of the same size.",
        "pingpong": "The set pingpongs: it plays to the last frame and then back down the same "
        "frames to frame 0. The first and last frames are the two ends of one movement, so "
        "neither has to meet the other, and no frame between them may repeat its neighbour.",
    }.get(plan.playback, "The set plays once and stops.")
    if plan.holds_final_frame:
        ending += " The last frame is a settled resting state the character holds."
    return (
        f"Character build: {spec.description[:400]}\n\n"
        f"Set: {set_name}\n"
        f"Intent: {intent}\n\n"
        f"{plan.frame_count} frames at {plan.fps} fps, drawn in the {plan.view} view. {ending}"
    )


def parse(reply: str) -> dict:
    """Pull the JSON object out of a reply that may be fenced or padded."""
    text = reply.strip()
    fence = re.search(r"```(?:json)?\s*(.+?)```", text, re.S)
    if fence:
        text = fence.group(1).strip()
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < 0:
        raise MotionError("the motion director returned no JSON object")
    try:
        return json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        raise MotionError(f"the motion director returned unreadable JSON: {exc}") from exc


def write_motion(
    model: BaseChatModel,
    spec: CharacterSpec,
    set_name: str,
    intent: str,
    plan: SetPlan,
    dst: Path | str,
    attempts: int = 2,
) -> MotionSheet:
    """Write a sheet for `set_name` to `dst` and return it.

    An existing sheet is reused, so a rerun costs nothing and every set keeps
    the plan its frames were actually drawn against.
    """
    dst = Path(dst)
    if dst.exists():
        return load_motion(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    last_error = ""
    for _ in range(attempts):
        reply = model.invoke(
            [SystemMessage(SYSTEM), HumanMessage(request(spec, set_name, intent, plan))]
        )
        try:
            written = parse(str(reply.content))
            sheet = {
                "name": set_name,
                "fps": plan.fps,
                "view": plan.view,
                "playback": plan.playback,
                "arc": str(written.get("arc", "")).strip(),
                "frames": [
                    {
                        "i": index,
                        "role": frame.get("role", "inbetween"),
                        "pace": frame.get("pace", "steady"),
                        "cue": str(frame.get("cue", "")).strip(),
                    }
                    for index, frame in enumerate(written["frames"][: plan.frame_count])
                ],
            }
            if len(sheet["frames"]) != plan.frame_count:
                raise MotionError(
                    f"asked for {plan.frame_count} frames, got {len(sheet['frames'])}"
                )
            dst.write_text(json.dumps(sheet, indent=2) + "\n")
            return load_motion(dst)  # validated the same as a traced sheet
        except (MotionError, KeyError, TypeError) as error:
            dst.unlink(missing_ok=True)
            last_error = str(error)
    raise MotionError(f"could not write a motion sheet for {set_name}: {last_error}")
