import subprocess
from pathlib import Path

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
    assert "#FF00FF" in seen["prompt"] and "art.png" in seen["prompt"]


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
    Image.new("RGB", (8, 8)).save(existing)

    def explode(*args, **kwargs):
        raise AssertionError("an existing render must not be drawn again")

    monkeypatch.setattr(subprocess, "run", explode)
    assert draw("a singer", existing) == existing


def test_draw_refuses_to_overwrite_when_reuse_is_off(tmp_path):
    existing = tmp_path / "art.png"
    Image.new("RGB", (8, 8)).save(existing)
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
    assert "#FF00FF" in record
    assert str(reference.resolve()) in record


def test_the_record_says_so_when_nothing_was_referenced(monkeypatch, tmp_path):
    monkeypatch.setattr(subprocess, "run", _writes())
    draw("a singer", tmp_path / "art.png")
    assert "(none)" in (tmp_path / "art.txt").read_text()
