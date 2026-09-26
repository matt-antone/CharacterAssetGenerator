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
        comfy, "Client", lambda **kw: real(api_key="k", transport=httpx.MockTransport(fake), **kw)
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


class FakeLocal(FakeCloud):
    """A ComfyUI server of your own: the same routes, and the image served directly."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/view":
            self.requests.append(request)
            return httpx.Response(200, content=self.image)
        return super().__call__(request)


@pytest.fixture
def local(monkeypatch, tmp_path):
    fake = FakeLocal()
    real = comfy.Client
    monkeypatch.setattr(comfy, "POLL_SECONDS", 0)
    monkeypatch.setattr(
        comfy, "Client", lambda **kw: real(transport=httpx.MockTransport(fake), **kw)
    )
    workflow = tmp_path / "workflow.json"
    workflow.write_text(json.dumps(WORKFLOW))
    fake.workflow = workflow
    return fake


def test_draw_with_local_sends_no_key_to_your_own_server(local, tmp_path, monkeypatch):
    monkeypatch.setenv("COMFY_CLOUD_API_KEY", "a-cloud-key-that-must-stay-home")
    monkeypatch.delenv("CAG_LOCAL_COMFY_URL", raising=False)
    out = draw("a singer", tmp_path / "art.png", backend="local", workflow=local.workflow)

    with Image.open(out) as image:
        assert image.size == (8, 8)
    assert {r.url.host for r in local.requests} == {"127.0.0.1"}
    assert all("x-api-key" not in r.headers for r in local.requests)
    assert "extra_data" not in local.submitted[0]


def test_a_local_error_says_it_was_local(tmp_path):
    def refuse(request):
        return httpx.Response(400, text="Cannot execute because node GeminiImage2Node does not exist.")

    client = comfy.Client(local=True, transport=httpx.MockTransport(refuse))
    with pytest.raises(comfy.ComfyError, match="local ComfyUI submit failed.*GeminiImage2Node"):
        client.submit({"1": {"class_type": "GeminiImage2Node", "inputs": {}}})


def test_a_local_build_needs_no_key_and_has_its_own_default_workflow(monkeypatch):
    from cag import cli

    monkeypatch.delenv("COMFY_CLOUD_API_KEY", raising=False)
    monkeypatch.delenv("COMFY_API_KEY", raising=False)
    monkeypatch.delenv("CAG_LOCAL_COMFY_WORKFLOW", raising=False)
    assert cli.draw_with("local") is not None
    # Nano Banana is a partner node: no local server can run the cloud default.
    assert comfy.workflow_path(local=True) == comfy.LOCAL_WORKFLOW != comfy.DEFAULT_WORKFLOW
    classes = {n["class_type"] for n in comfy.load_workflow(comfy.LOCAL_WORKFLOW).values()}
    assert not any(c.startswith("Gemini") for c in classes)


def test_every_backend_is_offered_and_none_goes_missing(capsys):
    """The Comfy work once replaced a backend instead of adding beside it."""
    import typing

    from cag import cli
    from cag.draw import BACKENDS, Backend

    assert set(BACKENDS) == set(typing.get_args(Backend))
    assert {"codex", "comfy", "local"} <= set(BACKENDS)
    with pytest.raises(SystemExit):
        cli.main(["build", "--help"])
    assert "{" + ",".join(BACKENDS) + "}" in capsys.readouterr().out


def test_extra_fills_placeholders_of_the_callers_own_and_wins():
    video = {**WORKFLOW, "9": {"class_type": "WanSCAILToVideo",
                               "inputs": {"length": "$length", "width": "$width"}}}
    filled = comfy.fill(video, "a dance", ["a"], 7, comfy.PORTRAIT,
                        extra={"$length": 41, "$width": 576})
    assert filled["9"]["inputs"] == {"length": 41, "width": 576}
    assert filled["7"]["inputs"]["width"] == 576, "extra is applied last"


def test_a_placeholder_nothing_fills_is_refused_before_it_is_sent():
    video = {**WORKFLOW, "9": {"class_type": "WanSCAILToVideo", "inputs": {"length": "$length"}}}
    with pytest.raises(comfy.ComfyError, match=r"9\.length=\$length"):
        comfy.fill(video, "a dance", ["a"], 7, comfy.PORTRAIT)
    with pytest.raises(comfy.ComfyError, match="must start with"):
        comfy.fill(WORKFLOW, "a dance", ["a"], 7, comfy.PORTRAIT, extra={"length": 41})
    # Only the workflow's own strings are placeholders: a prompt may start with $.
    filled = comfy.fill(WORKFLOW, "$5 of glitter", ["a"], 7, comfy.PORTRAIT)
    assert filled["6"]["inputs"]["prompt"] == "$5 of glitter"


@pytest.mark.parametrize("path", [
    comfy.DEFAULT_WORKFLOW, comfy.LOCAL_WORKFLOW, comfy.POSE_WORKFLOW,
    Path("comfy/qwen21-mannequin-pose.json"),
])
def test_every_workflow_a_render_sends_today_fills_with_nothing_left_over(path):
    shipped = comfy.load_workflow(path)
    names = [f"ref{n}" for n in range(comfy.reference_slots(shipped))]
    filled = comfy.fill(shipped, "a singer", names, 7, comfy.PORTRAIT)
    assert '"$' not in json.dumps(filled)


def test_saved_images_are_every_saved_one_in_filename_order():
    outputs = {
        "9": {"images": [{"filename": "p_0001.png", "type": "temp"}]},
        "8": {"images": [{"filename": f"cag_{n:05d}_.png", "type": "output"} for n in (3, 1, 2)]},
    }
    assert [i["filename"] for i in comfy.saved_images(outputs)] == [
        "cag_00001_.png", "cag_00002_.png", "cag_00003_.png"
    ]
    assert comfy.first_image(outputs)["filename"] == "cag_00001_.png"
    # Previews only when nothing was saved.
    assert [i["filename"] for i in comfy.saved_images({"9": outputs["9"]})] == ["p_0001.png"]
    assert comfy.saved_images({}) == []
    with pytest.raises(comfy.ComfyError, match="Save Image"):
        comfy.first_image({})


def apng(frames: int) -> bytes:
    buffer = io.BytesIO()
    first, *rest = [Image.new("RGB", (8, 12), (n * 40, 0, 0)) for n in range(frames)]
    first.save(buffer, format="PNG", save_all=True, append_images=rest, duration=62)
    return buffer.getvalue()


class FakeServer:
    """A ComfyUI that runs jobs saving many images, for `render_frames`."""

    def __init__(self, saves=3):
        self.saves = saves
        self.requests = []
        self.uploads = {}
        self.submitted = []
        self.status = {}
        self.object_info = {}

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/upload/image":
            body = request.read()
            name = body.split(b'filename="')[1].split(b'"')[0].decode()
            start = body.index(b"\r\n\r\n", body.index(b'filename="')) + 4
            self.uploads[name] = body[start : body.rindex(b"\r\n--")]
            return httpx.Response(200, json={"name": name})
        if path == "/api/prompt":
            self.submitted.append(json.loads(request.content))
            job_id = f"job-{len(self.submitted)}"
            self.status[job_id] = "completed"
            return httpx.Response(200, json={"prompt_id": job_id})
        if path.startswith("/api/jobs/"):
            job_id = path.removeprefix("/api/jobs/")
            if job_id not in self.status:
                return httpx.Response(404, json={"error": "Job not found"})
            images = [{"filename": f"cag_{n:05d}_.png", "subfolder": "", "type": "output"}
                      for n in reversed(range(self.saves))]
            return httpx.Response(200, json={
                "id": job_id, "status": self.status[job_id], "outputs": {"8": {"images": images}},
            })
        if path == "/api/view":
            n = int(request.url.params["filename"].split("_")[1])
            return httpx.Response(200, content=png((n, 0, 0)))
        if path.startswith("/api/object_info/"):
            kind = path.removeprefix("/api/object_info/")
            known = {kind: self.object_info[kind]} if kind in self.object_info else {}
            return httpx.Response(200, json=known)
        return httpx.Response(404)

    def client(self, local=True):
        return comfy.Client(api_key=None if local else "k", local=local,
                            transport=httpx.MockTransport(self))


VIDEO = {
    "1": {"class_type": "LoadImage", "inputs": {"image": "$image1"}},
    "2": {"class_type": "LoadImage", "inputs": {"image": "$image2"}},
    "5": {"class_type": "WanSCAILToVideo",
          "inputs": {"prompt": "$prompt", "length": "$length", "seed": "$seed",
                     "ref": ["1", 0], "drive": ["2", 0]}},
    "8": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
}


@pytest.fixture
def video(tmp_path, monkeypatch):
    monkeypatch.setattr(comfy, "POLL_SECONDS", 0)
    key = tmp_path / "key.png"
    key.write_bytes(png())
    drive = tmp_path / "drive.png"
    drive.write_bytes(apng(3))
    return [key, drive]


def frames(server, tmp_path, references, **kw):
    kw = {"raw": [False, True], "extra": {"$length": 3}, "expect": 3,
          "pending": tmp_path / "out" / "pending.json", **kw}
    return comfy.render_frames("a dance", tmp_path / "out", references, VIDEO, 60,
                               client=server.client(), **kw)


def test_a_raw_upload_keeps_every_frame_of_an_animated_png(video):
    server = FakeServer()
    name = server.client().upload_raw(video[1])
    assert name.startswith("cag-") and server.uploads[name] == video[1].read_bytes()
    with Image.open(io.BytesIO(server.uploads[name])) as image:
        assert image.n_frames == 3


def test_render_frames_writes_every_image_in_order(video, tmp_path):
    server = FakeServer(saves=3)
    written = frames(server, tmp_path, video, seed=7)
    assert [p.name for p in written] == ["000.png", "001.png", "002.png"]
    assert [Image.open(p).getpixel((0, 0))[0] for p in written] == [0, 1, 2]
    sent = server.submitted[0]["prompt"]
    assert sent["5"]["inputs"]["length"] == 3 and sent["5"]["inputs"]["seed"] == 7
    uploaded = server.uploads[sent["2"]["inputs"]["image"]]
    assert Image.open(io.BytesIO(uploaded)).n_frames == 3, "the drive went up raw"
    assert not (tmp_path / "out" / "pending.json").exists(), "done, so nothing to resume"


def test_render_frames_refuses_a_wrong_count_and_keeps_what_came_back(video, tmp_path):
    server = FakeServer(saves=2)
    with pytest.raises(comfy.ComfyError, match="saved 2 images, not the 3"):
        frames(server, tmp_path, video)
    with pytest.raises(comfy.ComfyError, match="rejected-1"):
        frames(server, tmp_path, video)
    assert sorted(p.name for p in (tmp_path / "out" / "rejected-0").iterdir()) == [
        "000.png", "001.png"
    ]
    assert not (tmp_path / "out" / "000.png").exists()
    assert not (tmp_path / "out" / "pending.json").exists(), "a finished job is not resumed"


def test_render_frames_resumes_a_running_job_instead_of_paying_again(video, tmp_path):
    server = FakeServer()
    server.status["job-9"] = "in_progress"
    pending = tmp_path / "out" / "pending.json"
    pending.parent.mkdir()
    pending.write_text(json.dumps({"job_id": "job-9"}))
    polls = []

    def finish(request):
        if request.url.path == "/api/jobs/job-9":
            polls.append(request)
            if len(polls) == 3:
                server.status["job-9"] = "completed"
        return server(request)

    client = comfy.Client(local=True, transport=httpx.MockTransport(finish))
    written = comfy.render_frames("a dance", tmp_path / "out", video, VIDEO, 60, raw=[False, True],
                                  extra={"$length": 3}, expect=3, pending=pending, client=client)
    assert len(written) == 3
    assert server.submitted == [] and server.uploads == {}
    assert not any(r.url.path == "/api/queue" for r in server.requests)


def test_render_frames_downloads_a_job_that_finished_while_nobody_waited(video, tmp_path):
    server = FakeServer()
    server.status["job-9"] = "completed"
    pending = tmp_path / "out" / "pending.json"
    pending.parent.mkdir()
    pending.write_text(json.dumps({"job_id": "job-9"}))
    assert len(frames(server, tmp_path, video)) == 3
    assert server.submitted == []


@pytest.mark.parametrize("known", [False, True])
def test_render_frames_submits_again_when_the_job_is_gone_or_failed(video, tmp_path, known):
    server = FakeServer()
    if known:
        server.status["job-9"] = "failed"
    pending = tmp_path / "out" / "pending.json"
    pending.parent.mkdir()
    pending.write_text(json.dumps({"job_id": "job-9"}))
    assert len(frames(server, tmp_path, video)) == 3
    assert len(server.submitted) == 1


def test_the_pending_job_is_written_before_the_wait(video, tmp_path, monkeypatch):
    server = FakeServer()
    pending = tmp_path / "out" / "pending.json"
    seen = {}

    def stopped(self, job_id, timeout, cancel=True):
        seen["pending"] = json.loads(pending.read_text())
        seen["cancel"] = cancel
        raise KeyboardInterrupt

    monkeypatch.setattr(comfy.Client, "wait", stopped)
    with pytest.raises(KeyboardInterrupt):
        frames(server, tmp_path, video)
    assert seen["pending"]["job_id"] == "job-1" and seen["cancel"] is False
    assert pending.exists(), "a build stopped mid-job resumes it"


@pytest.mark.parametrize("cancel", [True, False])
def test_a_wait_that_times_out_cancels_only_when_asked(monkeypatch, cancel):
    server = FakeServer()
    server.status["job-9"] = "in_progress"
    monkeypatch.setattr(comfy, "POLL_SECONDS", 0)
    with pytest.raises(comfy.ComfyError, match="timed out" if cancel else "left running"):
        server.client().wait("job-9", 0, cancel=cancel)
    assert any(r.url.path == "/api/queue" for r in server.requests) == cancel


def test_a_job_the_server_never_heard_of_is_none():
    assert FakeServer().client().job("job-404") is None


def test_render_uses_the_seed_it_is_given(cloud, tmp_path):
    comfy.render("a singer", tmp_path / "art.png", [], WORKFLOW, 60, client=comfy.Client(), seed=7)
    assert cloud.submitted[0]["prompt"]["6"]["inputs"]["seed"] == 7


def loader(name, files, v3=False):
    return {"input": {"required": {name: ("COMBO", {"options": files}) if v3 else (files,)}}}


def test_preflight_lists_every_missing_node_and_model_file(tmp_path):
    server = FakeServer()
    server.object_info = {
        "UnetLoaderGGUF": loader("unet_name", ["SCAIL-2-Q5_K_M.gguf"]),
        "LoraLoaderModelOnly": loader("lora_name", ["dpo.safetensors"], v3=True),
        "CLIPLoader": loader("clip_name", ["umt5.safetensors"]),
        "LoadImage": loader("image", ["a.png"]),
        "SaveImage": {"input": {"required": {}}},
    }
    video_graph = tmp_path / "video.json"
    video_graph.write_text(json.dumps({
        "1": {"class_type": "UnetLoaderGGUF", "inputs": {"unet_name": "SCAIL-2-Q2_K.gguf"}},
        "2": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "lightx2v.safetensors",
                                                             "model": ["1", 0]}},
        "3": {"class_type": "LoraLoaderModelOnly", "inputs": {"lora_name": "dpo.safetensors",
                                                             "model": ["2", 0]}},
        "4": {"class_type": "LoadImage", "inputs": {"image": "$image1"}},
        "5": {"class_type": "WanSCAILToVideo", "inputs": {"prompt": "$prompt"}},
        "6": {"class_type": "SaveImage", "inputs": {"images": ["5", 0]}},
    }))
    restyle = {"c": {"class_type": "CLIPLoader", "inputs": {"clip_name": "qwen3vl.safetensors"}}}
    missing = comfy.preflight(server.client(), [video_graph, restyle])
    assert missing == [
        "video.json: UnetLoaderGGUF unet_name SCAIL-2-Q2_K.gguf is not installed",
        "video.json: LoraLoaderModelOnly lora_name lightx2v.safetensors is not installed",
        "video.json: node WanSCAILToVideo is not installed",
        "workflow 2: CLIPLoader clip_name qwen3vl.safetensors is not installed",
    ]
    asked = [r.url.path for r in server.requests]
    assert asked.count("/api/object_info/LoraLoaderModelOnly") == 1, "each class asked once"


def test_preflight_on_a_server_that_cannot_answer_checks_nothing():
    def cloud(request):
        return httpx.Response(404)

    client = comfy.Client(api_key="k", transport=httpx.MockTransport(cloud))
    workflow = {"1": {"class_type": "UNETLoader", "inputs": {"unet_name": "x"}}}
    assert comfy.preflight(client, [workflow]) == []
