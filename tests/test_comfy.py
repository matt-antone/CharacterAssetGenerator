import io
import json
from pathlib import Path

import httpx
import pytest
from PIL import Image

from cag import comfy
from cag.draw import DrawError, draw
from cag.spec import load_spec

#: The shape of a Nano Banana workflow: three references batched into one input.
WORKFLOW = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "$image1"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "$image2"}},
    "3": {"class_type": "LoadImage", "inputs": {"image": "$image3"}},
    "4": {"class_type": "ImageBatch", "inputs": {"image1": ["1", 0], "image2": ["2", 0]}},
    "5": {"class_type": "ImageBatch", "inputs": {"image1": ["4", 0], "image2": ["3", 0]}},
    "6": {
        "class_type": "GeminiImageNode",
        "inputs": {"prompt": "$prompt", "seed": "$seed", "images": ["5", 0]},
    },
    "7": {"class_type": "EmptyLatentImage", "inputs": {"width": "$width", "height": "$height"}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["6", 0], "filename_prefix": "cag"}},
}


def png(colour=(255, 0, 255), size=(8, 8)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, colour).save(buffer, format="PNG")
    return buffer.getvalue()


SHEET = comfy.Canvas(1536, 1024, "3:2")


def test_fill_puts_the_prompt_seed_canvas_and_references_in_place():
    filled = comfy.fill(WORKFLOW, "a singer", ["a.png", "b.png", "c.png"], 7, SHEET)
    assert filled["6"]["inputs"]["prompt"] == "a singer"
    assert filled["6"]["inputs"]["seed"] == 7
    assert filled["7"]["inputs"] == {"width": 1536, "height": 1024}
    assert [filled[n]["inputs"]["image"] for n in "123"] == ["a.png", "b.png", "c.png"]
    assert WORKFLOW["6"]["inputs"]["prompt"] == "$prompt", "the loaded workflow is reused"


def test_unused_reference_slots_are_dropped_and_their_batches_bypassed():
    filled = comfy.fill(WORKFLOW, "a singer", ["a.png"], 7, comfy.PORTRAIT)
    assert set(filled) == {"1", "6", "7", "8"}
    assert filled["6"]["inputs"]["images"] == ["1", 0]


def test_a_render_with_no_references_leaves_the_image_input_unset():
    filled = comfy.fill(WORKFLOW, "a room", [], 7, SHEET)
    assert set(filled) == {"6", "7", "8"}
    assert "images" not in filled["6"]["inputs"]


def test_more_references_than_slots_is_refused():
    with pytest.raises(comfy.ComfyError, match=r"4 reference images .* 3 slots"):
        comfy.fill(WORKFLOW, "a singer", ["a", "b", "c", "d"], 7, comfy.PORTRAIT)


def test_the_shipped_workflow_takes_every_reference_a_render_sends():
    shipped = comfy.load_workflow(comfy.DEFAULT_WORKFLOW)
    # Key art, carried frame, detail, pose grid: the most any render attaches.
    assert comfy.reference_slots(shipped) == 4
    full = comfy.fill(shipped, "a singer", ["k", "c", "d", "p"], 7, SHEET)
    assert [full["5"]["inputs"][f"images.image{n}"] for n in range(4)] == [
        ["1", 0], ["2", 0], ["3", 0], ["4", 0]
    ], "the pose grid stays the last reference, as the prompt says"
    assert full["6"]["inputs"]["aspect_ratio"] == "3:2"

    two = comfy.fill(shipped, "a singer", ["k", "p"], 7, comfy.PORTRAIT)
    assert set(two["5"]["inputs"]) == {"images.image0", "images.image1"}
    assert two["6"]["inputs"]["aspect_ratio"] == "2:3"

    one = comfy.fill(shipped, "a singer", ["k"], 7, comfy.PORTRAIT)
    assert "5" not in one and one["6"]["inputs"]["images"] == ["1", 0]

    none = comfy.fill(shipped, "a room", [], 7, comfy.PORTRAIT)
    assert set(none) == {"6", "7"} and "images" not in none["6"]["inputs"]


def test_the_shipped_pose_workflow_takes_the_character_and_one_pose():
    from cag.prompts import POSE_EDIT

    shipped = comfy.load_workflow(comfy.POSE_WORKFLOW)
    assert comfy.reference_slots(shipped) == 2
    assert comfy.canvas(POSE_EDIT) == comfy.PORTRAIT, "one figure, full height"
    filled = comfy.fill(shipped, POSE_EDIT, ["key", "pose"], 7, comfy.PORTRAIT)
    assert [filled["9"]["inputs"]["image"], filled["10"]["inputs"]["image"]] == ["key", "pose"]
    assert filled["13"]["inputs"]["width"] == 1024 and filled["13"]["inputs"]["height"] == 1536
    assert "$" not in json.dumps(filled), "every placeholder is filled"


def test_a_workflow_that_batches_nothing_gets_its_images_unpadded(cloud, tmp_path, monkeypatch):
    """Letterboxing only exists to survive a batch."""
    monkeypatch.setattr(comfy, "letterbox", lambda path: pytest.fail("letterboxed"))
    pose_only = {
        "1": {"class_type": "LoadImage", "inputs": {"image": "$image1"}},
        "2": {"class_type": "Edit", "inputs": {"prompt": "$prompt", "image": ["1", 0]}},
        "8": {"class_type": "SaveImage", "inputs": {"images": ["2", 0]}},
    }
    workflow = tmp_path / "pose.json"
    workflow.write_text(json.dumps(pose_only))
    reference = tmp_path / "key.png"
    reference.write_bytes(png())
    draw("a singer", tmp_path / "art.png", references=[reference], backend="comfy", workflow=workflow)
    assert cloud.submitted


def test_each_prompt_gets_the_canvas_it_asks_for():
    from cag.prompts import FRAME_VIEWS, frame_sheet_prompt, location_prompt

    spec = load_spec("tests/fixtures/velvet-lou.json")
    sheet = frame_sheet_prompt(spec, "B", "N", FRAME_VIEWS["front"], [("key", "step")])
    assert comfy.canvas(sheet).aspect == "3:2"
    # A set drawn whole, eight across, needs the width for hair and arms.
    whole = frame_sheet_prompt(spec, "B", "N", FRAME_VIEWS["front"], [("key", "step")] * 16, per_row=8)
    assert comfy.canvas(whole).aspect == "21:9"
    eight = frame_sheet_prompt(spec, "B", "N", FRAME_VIEWS["front"], [("key", "step")] * 8)
    assert comfy.canvas(eight).aspect == "3:2", "codex's batches of eight are unchanged"
    assert comfy.canvas(location_prompt(spec)).aspect == "16:9"
    assert comfy.canvas("Draw Lou standing.") == comfy.PORTRAIT


def test_a_reference_is_letterboxed_whole_in_its_own_border_colour(tmp_path):
    grid = tmp_path / "grid.png"
    image = Image.new("RGB", (300, 200), (255, 0, 255))
    image.paste((0, 0, 0), (0, 90, 300, 110))  # a band edge to edge
    image.save(grid)
    with Image.open(io.BytesIO(comfy.letterbox(grid))) as boxed:
        side = comfy.REFERENCE_SIDE
        assert boxed.size == (side, side)
        assert boxed.getpixel((5, 5)) == (255, 0, 255), "bars match the backdrop"
        # Nothing cropped: the band still reaches both side edges.
        assert boxed.getpixel((0, side // 2)) == boxed.getpixel((side - 1, side // 2)) == (0, 0, 0)


def test_panel_rules_are_painted_out_and_figures_are_not():
    import numpy

    magenta = (255, 0, 255)
    sheet = Image.new("RGB", (400, 300), magenta)
    for y in (0, 148, 294):  # rules along the top, across the middle, along the bottom
        sheet.paste((0, 0, 0), (0, y, 400, y + 6))
    sheet.paste((0, 0, 0), (98, 0, 104, 300))  # a rule the full height
    sheet.paste((0, 0, 0), (298, 0, 304, 152))  # one spanning the top row of panels only
    # A full-height rule whose anti-aliased edge column is dark only from 40% down,
    # as on Belter's whole-set sing sheet: its extent is the whole band's.
    sheet.paste((0, 0, 0), (351, 0, 357, 300))
    sheet.paste((0, 0, 0), (350, 120, 351, 300))
    # A figure's dark prop: long and straight, but it ends in open backdrop.
    sheet.paste((10, 10, 10), (200, 30, 206, 140))

    cleaned, erased = comfy.erase_panel_rules(sheet)
    assert erased == 6
    pixels = numpy.array(cleaned)
    assert (pixels[:8].reshape(-1, 3) == magenta).all(), "the top rule is gone"
    assert (pixels[:, 96:106].reshape(-1, 3) == magenta).all(), "the full-height rule is gone"
    assert tuple(pixels[80, 300]) == magenta, "the half-height rule is gone"
    assert (pixels[:, 348:360].reshape(-1, 3) == magenta).all(), "the anti-aliased rule is gone"
    assert tuple(pixels[80, 203]) == (10, 10, 10), "the prop is kept"


def test_a_clean_render_is_left_exactly_as_drawn():
    # A figure in black leggings: a tall dark shape, clear of every edge.
    sheet = Image.new("RGB", (600, 400), (255, 0, 255))
    sheet.paste((0, 0, 0), (250, 60, 270, 340))
    cleaned, erased = comfy.erase_panel_rules(sheet)
    assert erased == 0 and cleaned is sheet


def test_a_workflow_needs_somewhere_to_put_the_prompt(tmp_path):
    missing = tmp_path / "workflow.json"
    with pytest.raises(comfy.ComfyError, match="Export"):
        comfy.load_workflow(missing)
    missing.write_text(json.dumps({"nodes": [], "links": []}))
    with pytest.raises(comfy.ComfyError, match="UI-format"):
        comfy.load_workflow(missing)
    missing.write_text(json.dumps({"1": {"class_type": "SaveImage", "inputs": {}}}))
    with pytest.raises(comfy.ComfyError, match=r"\$prompt"):
        comfy.load_workflow(missing)


class FakeCloud:
    """Comfy Cloud, as far as a render sees it."""

    def __init__(self, image=None, fail=False):
        self.image = image or png()
        self.fail = fail
        self.requests = []
        self.submitted = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/upload/image":
            return httpx.Response(200, json={"name": "cag-ref.png", "subfolder": ""})
        if path == "/api/prompt":
            self.submitted.append(json.loads(request.content))
            return httpx.Response(200, json={"prompt_id": "job-1"})
        if path == "/api/jobs/job-1":
            if self.fail:
                return httpx.Response(200, json={"status": "failed", "execution_error": "OOM"})
            polls = sum(r.url.path == path for r in self.requests)
            if polls == 1:
                return httpx.Response(200, json={"status": "in_progress"})
            return httpx.Response(200, json={
                "status": "completed",
                "outputs": {"8": {"images": [
                    {"filename": "cag_0001.png", "subfolder": "", "type": "output"}
                ]}},
            })
        if path == "/api/view":
            return httpx.Response(302, headers={"location": "https://storage.example/signed"})
        if request.url.host == "storage.example":
            return httpx.Response(200, content=self.image)
        return httpx.Response(404)


@pytest.fixture
def cloud(monkeypatch, tmp_path):
    fake = FakeCloud()
    real = comfy.Client
    monkeypatch.setattr(comfy, "POLL_SECONDS", 0)
    monkeypatch.setattr(
        comfy, "Client", lambda: real(api_key="k", transport=httpx.MockTransport(fake))
    )
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(WORKFLOW))
    fake.workflow = workflow
    return fake


def test_draw_with_comfy_uploads_submits_waits_and_saves(cloud, tmp_path):
    reference = tmp_path / "key.png"
    reference.write_bytes(png((0, 0, 0)))
    out = draw("a singer", tmp_path / "art.png", references=[reference],
               backend="comfy", workflow=cloud.workflow)

    with Image.open(out) as image:
        assert image.format == "PNG" and image.size == (8, 8)
    sent = cloud.submitted[0]
    assert sent["prompt"]["6"]["inputs"]["prompt"] == "a singer"
    assert sent["prompt"]["1"]["inputs"]["image"] == "cag-ref.png"
    assert sent["extra_data"] == {"api_key_comfy_org": "k"}
    # The signed download URL is another host: it never sees the key.
    storage = next(r for r in cloud.requests if r.url.host == "storage.example")
    assert "x-api-key" not in storage.headers
    # A workflow is not an agent, so the record is the prompt and nothing else.
    record = (tmp_path / "art.txt").read_text()
    assert record.startswith("a singer\n") and "Generate exactly one image" not in record


def test_a_failed_job_is_retried_then_reported(cloud, tmp_path):
    cloud.fail = True
    with pytest.raises(DrawError, match="failed: OOM"):
        draw("a singer", tmp_path / "art.png", backend="comfy", workflow=cloud.workflow)
    assert len(cloud.submitted) == 2


def test_a_render_with_scenery_behind_it_is_redrawn(cloud, tmp_path):
    stripes = Image.new("RGB", (64, 64))
    stripes.putdata([(x * 4, x * 4, x * 4) for _ in range(64) for x in range(64)])
    buffer = io.BytesIO()
    stripes.save(buffer, format="PNG")
    cloud.image = buffer.getvalue()
    with pytest.raises(DrawError, match="scenery"):
        draw("a singer", tmp_path / "art.png", backend="comfy", workflow=cloud.workflow)
    assert len(cloud.submitted) == 2
    assert not (tmp_path / "art.png").exists()


def test_too_many_references_fails_before_anything_is_sent(cloud, tmp_path):
    references = []
    for n in range(4):
        references.append(tmp_path / f"{n}.png")
        references[-1].write_bytes(png())
    with pytest.raises(DrawError, match="3 slots"):
        draw("a singer", tmp_path / "art.png", references=references,
             backend="comfy", workflow=cloud.workflow)
    assert cloud.requests == []


def test_a_build_without_a_key_stops_before_drawing(monkeypatch, tmp_path):
    from cag import cli

    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(WORKFLOW))
    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    with pytest.raises(SystemExit, match="COMFY_CLOUD_API_KEY"):
        cli.draw_with("comfy", workflow)
    assert cli.draw_with("codex") is None
    # The name Comfy's own samples use works too.
    monkeypatch.setenv("COMFY_API_KEY", "k")
    assert cli.draw_with("comfy", workflow) is not None
