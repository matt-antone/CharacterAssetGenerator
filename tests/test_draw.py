import subprocess
from pathlib import Path

import numpy

import pytest
from PIL import Image

from cag.draw import DrawError, draw


def _writes(png: bool = True, code: int = 0):
    def fake_run(argv, **kwargs):
        if code == 0 and png:
            out = Path(argv[argv.index("--cd") + 1]) / "art.png"
            Image.new("RGB", (8, 8), (255, 0, 255)).save(out)
        return subprocess.CompletedProcess(argv, code, "", "boom" if code else "")

    return fake_run


def test_draw_returns_verified_png(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _writes())
    out = draw("a singer", tmp_path / "art.png")
    assert out.exists()
    with Image.open(out) as image:
        assert image.size == (8, 8)


def test_draw_attaches_references(monkeypatch, tmp_path):
    reference = tmp_path / "key.png"
    Image.new("RGB", (4, 4)).save(reference)
    seen = {}

    def capture(argv, **kwargs):
        seen["argv"] = argv
        seen["prompt"] = kwargs["input"]
        return _writes()(argv, **kwargs)

    monkeypatch.setattr(subprocess, "run", capture)
    draw("a singer", tmp_path / "art.png", references=[reference])
    assert ["--image", str(reference.resolve())] == seen["argv"][-3:-1]
    # draw only handles the file; the look and the backdrop belong to the prompt.
    assert "a singer" in seen["prompt"] and "art.png" in seen["prompt"]


def test_draw_retries_then_fails_when_no_file(monkeypatch, tmp_path):
    calls = []

    def never_writes(argv, **kwargs):
        calls.append(argv)
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", never_writes)
    with pytest.raises(DrawError, match="wrote no file"):
        draw("a singer", tmp_path / "art.png")
    assert len(calls) == 2


def test_draw_reuses_a_finished_render_so_a_run_can_resume(monkeypatch, tmp_path):
    existing = tmp_path / "art.png"
    Image.new("RGB", (8, 8), (255, 0, 255)).save(existing)

    def explode(*args, **kwargs):
        raise AssertionError("an existing render must not be drawn again")

    monkeypatch.setattr(subprocess, "run", explode)
    assert draw("a singer", existing) == existing


def test_draw_refuses_to_overwrite_when_reuse_is_off(tmp_path):
    existing = tmp_path / "art.png"
    Image.new("RGB", (8, 8), (255, 0, 255)).save(existing)
    with pytest.raises(DrawError, match="refusing to overwrite"):
        draw("a singer", existing, reuse=False)


def test_draw_rejects_unreadable_output(monkeypatch, tmp_path):
    def writes_junk(argv, **kwargs):
        (Path(argv[argv.index("--cd") + 1]) / "art.png").write_bytes(b"not a png")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", writes_junk)
    with pytest.raises(DrawError, match="not a readable image"):
        draw("a singer", tmp_path / "art.png")


def test_draw_records_the_prompt_and_references_it_sent(monkeypatch, tmp_path):
    reference = tmp_path / "key.png"
    Image.new("RGB", (4, 4)).save(reference)
    monkeypatch.setattr(subprocess, "run", _writes())
    draw("a singer in green", tmp_path / "art.png", references=[reference])
    record = (tmp_path / "art.txt").read_text()
    assert "a singer in green" in record
    assert "art.png" in record
    assert str(reference.resolve()) in record


def test_the_record_says_so_when_nothing_was_referenced(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _writes())
    draw("a singer", tmp_path / "art.png")
    assert "(none)" in (tmp_path / "art.txt").read_text()


def scenery(size=(64, 64)):
    """A render with a busy background instead of a flat backdrop."""
    pixels = numpy.tile(
        numpy.linspace(0, 255, size[0], dtype=numpy.uint8)[None, :, None], (size[1], 1, 3)
    )
    return Image.fromarray(pixels, "RGB")


def test_a_render_with_scenery_behind_it_is_redrawn_not_kept(monkeypatch, tmp_path):
    attempts = []

    def draws_scenery(argv, **kwargs):
        attempts.append(argv)
        scenery().save(Path(argv[argv.index("--cd") + 1]) / "art.png")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(subprocess, "run", draws_scenery)
    with pytest.raises(DrawError, match="scenery behind the character"):
        draw("a singer", tmp_path / "art.png")
    assert len(attempts) == 2
    assert not (tmp_path / "art.png").exists()
    # Each unusable attempt is kept aside to be looked at, never resumed from.
    assert sorted(p.name for p in tmp_path.glob("art.unusable-*.png")) == [
        "art.unusable-0.png", "art.unusable-1.png"
    ]


def test_verification_never_deletes_an_existing_render(tmp_path):
    """A resume must not destroy work it decides it cannot use."""
    from cag.draw import _verify

    existing = tmp_path / "art.png"
    scenery().save(existing)
    with pytest.raises(DrawError):
        _verify(existing)
    assert existing.exists()


def test_the_hue_is_free_but_the_backdrop_must_clear_the_outline():
    """Vision does not chroma key, so any colour works — except the outline's."""
    from cag.draw import backdrop_is_usable

    for colour in ((255, 0, 255), (236, 18, 222), (0, 200, 0), (90, 140, 255)):
        assert backdrop_is_usable(Image.new("RGB", (64, 64), colour))[0], colour


def test_a_backdrop_the_colour_of_the_outline_is_refused():
    """Black backdrop plus black outline is unmaskable: they are one colour."""
    from cag.draw import backdrop_is_usable

    for colour in ((0, 0, 0), (18, 20, 24)):
        usable, why_not = backdrop_is_usable(Image.new("RGB", (64, 64), colour))
        assert not usable
        assert "too close to the black outline" in why_not


def test_scenery_is_refused_separately_from_colour():
    from cag.draw import backdrop_is_usable

    usable, why_not = backdrop_is_usable(scenery())
    assert not usable
    assert "scenery behind the character" in why_not


def test_a_washed_out_magenta_is_refused():
    """The key cuts costume away against an orchid backdrop, so it is drawn again."""
    from cag.draw import backdrop_is_usable

    # What Nano Banana painted behind two of Belter's frame sheets.
    for colour in ((194, 80, 159), (203, 62, 184)):
        usable, why_not = backdrop_is_usable(Image.new("RGB", (64, 64), colour))
        assert not usable
        assert "washed-out magenta" in why_not
    # What it painted behind the rest, and what the prompt asks for.
    for colour in ((252, 3, 250), (255, 0, 255)):
        assert backdrop_is_usable(Image.new("RGB", (64, 64), colour))[0], colour


def test_a_fixed_seed_moves_on_for_every_redraw_even_across_runs(monkeypatch, tmp_path):
    """A fixed seed that failed once would fail again; each try is the next seed on."""
    from cag import draw as drawing

    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"1": {"class_type": "Edit", "inputs": {"prompt": "$prompt"}}}')
    seeds, pictures = [], []

    def run(prompt, out_path, references, loaded, timeout, scene, local, seed=None):
        seeds.append(seed)
        (pictures.pop(0) if pictures else scenery()).save(out_path)
        return ""

    monkeypatch.setattr(drawing, "_run_comfy", run)
    out = tmp_path / "art.png"
    with pytest.raises(DrawError):
        draw("a singer", out, backend="comfy", workflow=workflow, seed=100)
    first = (tmp_path / "art.unusable-0.png").read_bytes()
    with pytest.raises(DrawError):
        draw("a singer", out, backend="comfy", workflow=workflow, seed=100)
    assert seeds == [100, 101, 102, 103]
    # Evidence from the first run is kept, not overwritten by the second.
    assert sorted(p.name for p in tmp_path.glob("art.unusable-*.png")) == [
        f"art.unusable-{n}.png" for n in range(4)
    ]
    assert (tmp_path / "art.unusable-0.png").read_bytes() == first

    pictures.append(Image.new("RGB", (64, 64), (255, 0, 255)))
    assert draw("a singer", out, backend="comfy", workflow=workflow, seed=100) == out
    assert seeds[-1] == 104, "unusable-N is always the picture seed + N made"


def test_without_a_seed_every_try_is_a_random_roll(monkeypatch, tmp_path):
    from cag import draw as drawing

    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"1": {"class_type": "Edit", "inputs": {"prompt": "$prompt"}}}')
    seeds = []

    def run(prompt, out_path, references, loaded, timeout, scene, local, seed=None):
        seeds.append(seed)
        scenery().save(out_path)
        return ""

    monkeypatch.setattr(drawing, "_run_comfy", run)
    (tmp_path / "art.unusable-0.png").write_bytes(b"from an earlier run")
    with pytest.raises(DrawError):
        draw("a singer", tmp_path / "art.png", backend="comfy", workflow=workflow)
    assert seeds == [None, None]
    assert (tmp_path / "art.unusable-0.png").read_bytes() == b"from an earlier run"
    assert (tmp_path / "art.unusable-2.png").exists()


def test_a_workflows_own_placeholders_reach_the_render(monkeypatch, tmp_path):
    """A machine profile's `$resolution` has no other way into a restyle."""
    from cag import draw as drawing

    workflow = tmp_path / "workflow.json"
    workflow.write_text('{"1": {"class_type": "Edit", "inputs": {"prompt": "$prompt"}}}')
    sent = []

    def run(prompt, out_path, references, loaded, timeout, scene, local, extra=None, seed=None):
        sent.append(extra)
        Image.new("RGB", (64, 64), (255, 0, 255)).save(out_path)
        return ""

    monkeypatch.setattr(drawing, "_run_comfy", run)
    draw("a singer", tmp_path / "art.png", backend="comfy", workflow=workflow,
         extra={"$resolution": 512})
    assert sent == [{"$resolution": 512}]
