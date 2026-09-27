"""`cag motions pull`: a bundle MotionArtist synced to Drive, installed checked.

rclone is never run: a fake stands in for it, copying out of a local directory
that plays the remote.
"""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from cag import cli, pull
from cag.motion import library, read_bundle
from cag.pull import PullError

REMOTE = "kadrive:MotionArtist"


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tracer_bundle(remote_root: Path, name="shuffle-3", set_name="shuffle", heads=None, **clip):
    """A bundle as MotionArtist syncs it: the sample, with its clip, mask and head boxes."""
    root = remote_root / set_name / name
    shutil.copytree("motions/sample", root)
    motion_sheet = json.loads((root / "motion.json").read_text())
    motion_sheet["name"] = name
    for frame in motion_sheet["frames"]:
        frame["clip_frame"] = 12 + 6 * frame["i"]
    (root / "motion.json").write_text(json.dumps(motion_sheet))
    (root / "clip.mp4").write_bytes(b"the clip the tracer traced")
    (root / "mask.mp4").write_bytes(b"white on black")
    (root / "heads.json").write_text(json.dumps(heads if heads is not None else [[300, 100, 420, 240]] * 120))
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["name"] = name
    manifest["files"] = {f: sha(root / f) for f in manifest["files"]}
    manifest["clip"] = {
        "file": "clip.mp4", "start": 15.5, "fps": 24.0, "frame_count": 120, "size": [720, 1280],
        "box": [0, 0, 720, 1280], "t_offset": 0.0, "sha256": sha(root / "clip.mp4"),
        "mask": {"file": "mask.mp4", "sha256": sha(root / "mask.mp4")},
        "heads": {"file": "heads.json", "sha256": sha(root / "heads.json")},
        **clip,
    }
    (root / "manifest.json").write_text(json.dumps(manifest))
    return root


def fake_rclone(remote_root: Path, calls: list, fail: str = ""):
    """A `subprocess.run` that answers rclone's `copy` and `lsf` from `remote_root`."""

    def run(argv, **kwargs):
        calls.append((argv, kwargs))
        if fail:
            return subprocess.CompletedProcess(argv, 3, "", fail)
        verb = argv[1]
        if verb == "copy":
            source = remote_root / argv[2].removeprefix(REMOTE).lstrip("/")
            if not source.is_dir():
                return subprocess.CompletedProcess(argv, 3, "", "error: directory not found")
            shutil.copytree(source, argv[3])
            return subprocess.CompletedProcess(argv, 0, "", "")
        if verb == "lsf":
            where = [a for a in argv[2:] if a.startswith(REMOTE)][0]
            base = remote_root / where.removeprefix(REMOTE).lstrip("/")
            if "-R" in argv:
                lines = sorted(
                    f"{p.relative_to(base).as_posix()}/" for p in base.rglob("*")
                    if p.is_dir() and len(p.relative_to(base).parts) <= 2
                )
            else:
                lines = sorted(f"{p.name}/" for p in base.iterdir() if p.is_dir())
            return subprocess.CompletedProcess(argv, 0, "\n".join(lines) + "\n", "")
        raise AssertionError(f"rclone {verb} was never meant to run")

    return run


def which(name):
    return f"/usr/bin/{name}"


def pulling(tmp_path, target="shuffle/shuffle-3", fail="", probe=None):
    calls = []
    done = pull.pull(
        target, tmp_path / "motions", REMOTE, fake_rclone(tmp_path / "remote", calls, fail), which, probe
    )
    return done, calls


def test_a_pull_installs_the_bundle_unrenamed_and_checked(tmp_path):
    tracer_bundle(tmp_path / "remote")
    done, calls = pulling(tmp_path)
    home = tmp_path / "motions" / "shuffle" / "shuffle-3"
    assert done.bundle.root == home and not done.replaced
    assert done.source == "kadrive:MotionArtist/shuffle/shuffle-3"
    bundle = library(tmp_path / "motions")["shuffle-3"]
    motion = bundle.load()
    assert motion.clip.mask is not None and motion.clip.heads is not None
    assert motion.clip.from_tracer and motion.clip_frames[:2] == (12, 18)
    # Nothing left beside motions/ once it is done.
    assert sorted(p.name for p in tmp_path.iterdir()) == ["motions", "remote"]


def test_rclone_is_bounded_and_never_reads_a_terminal(tmp_path):
    tracer_bundle(tmp_path / "remote")
    _, calls = pulling(tmp_path)
    (argv, kwargs), = calls
    assert argv[:2] == ["/usr/bin/rclone", "copy"] and argv[2] == "kadrive:MotionArtist/shuffle/shuffle-3"
    assert "--ask-password=false" in argv and "--retries" in argv
    assert kwargs["stdin"] is subprocess.DEVNULL and kwargs["timeout"] == pull.TIMEOUT


def test_a_re_pull_replaces_the_bundle_in_place(tmp_path):
    remote = tracer_bundle(tmp_path / "remote")
    pulling(tmp_path)
    home = tmp_path / "motions" / "shuffle" / "shuffle-3"
    (home / "stray.txt").write_text("left from before")
    (remote / "clip.mp4").write_bytes(b"the tracer re-cut it")
    manifest = json.loads((remote / "manifest.json").read_text())
    manifest["clip"]["sha256"] = sha(remote / "clip.mp4")
    (remote / "manifest.json").write_text(json.dumps(manifest))
    done, _ = pulling(tmp_path)
    assert done.replaced
    assert (home / "clip.mp4").read_bytes() == b"the tracer re-cut it"
    assert not (home / "stray.txt").exists(), "replaced, not merged"


@pytest.mark.parametrize("where", ["manifest", "motion.json"])
def test_a_bundle_whose_names_disagree_is_refused_and_nothing_is_installed(tmp_path, where):
    remote = tracer_bundle(tmp_path / "remote")
    file = remote / ("manifest.json" if where == "manifest" else "motion.json")
    data = json.loads(file.read_text())
    data["name"] = "shuffle-03"
    file.write_text(json.dumps(data))
    if where == "motion.json":
        manifest = json.loads((remote / "manifest.json").read_text())
        manifest["files"]["motion.json"] = sha(file)
        (remote / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(PullError, match="must be one name"):
        pulling(tmp_path)
    assert not (tmp_path / "motions" / "shuffle").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["motions", "remote"], "staging cleared"


def test_a_refused_re_pull_leaves_the_installed_copy_alone(tmp_path):
    remote = tracer_bundle(tmp_path / "remote")
    pulling(tmp_path)
    home = tmp_path / "motions" / "shuffle" / "shuffle-3"
    before = (home / "manifest.json").read_text()
    (remote / "mask.mp4").write_bytes(b"half uploaded")
    with pytest.raises(PullError, match="mask mask.mp4 is stale"):
        pulling(tmp_path)
    assert (home / "manifest.json").read_text() == before
    assert (home / "mask.mp4").read_bytes() == b"white on black"


def test_a_failed_rclone_says_why_and_installs_nothing(tmp_path):
    tracer_bundle(tmp_path / "remote")
    with pytest.raises(PullError, match="token needs renewing: rclone config reconnect kadrive:"):
        pulling(tmp_path, fail="couldn't fetch token: oauth2: invalid_grant")
    assert not (tmp_path / "motions" / "shuffle").exists()
    with pytest.raises(PullError, match="nothing at that path on the remote"):
        pulling(tmp_path, target="shuffle/shuffle-9")


def test_a_hung_rclone_is_refused(tmp_path):
    def hung(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs["timeout"])

    with pytest.raises(PullError, match="gave no answer"):
        pull.pull("shuffle/shuffle-3", tmp_path / "motions", REMOTE, hung, which)


@pytest.mark.parametrize("target", ["shuffle", "shuffle/club-3", "shuffle/shuffle", "shuffle/shuffle-x3"])
def test_a_target_that_is_not_set_slash_set_index_is_refused(tmp_path, target):
    with pytest.raises(PullError, match="not <set>/<set>-<index>"):
        pull.pull(target, tmp_path / "motions", REMOTE, fake_rclone(tmp_path, []), which)


def test_the_tracers_zero_padded_names_are_taken_as_written():
    assert pull.parse_target("club/club-01") == ("club", "club-01")
    assert pull.parse_target("shuffle/shuffle-3") == ("shuffle", "shuffle-3")


def test_a_set_with_a_hyphen_in_its_name_is_a_set(tmp_path):
    assert pull.parse_target("lindey-hop/lindey-hop-2") == ("lindey-hop", "lindey-hop-2")


@pytest.mark.parametrize("remote, said", [(None, "MOTION_ARTIST_REMOTE"), ("kadrive", "colon")])
def test_no_remote_or_a_bare_word_is_refused(tmp_path, remote, said):
    with pytest.raises(PullError, match=said):
        pull.pull("shuffle/shuffle-3", tmp_path / "motions", remote, fake_rclone(tmp_path, []), which)


def test_a_bundle_already_installed_under_another_directory_is_refused(tmp_path):
    tracer_bundle(tmp_path / "remote")
    tracer_bundle(tmp_path / "motions" / "elsewhere")
    with pytest.raises(PullError, match="already calls itself shuffle-3"):
        pulling(tmp_path)


def test_a_listed_file_that_did_not_arrive_is_refused(tmp_path):
    remote = tracer_bundle(tmp_path / "remote")
    (remote / "thumbs" / "f03.jpg").unlink()
    with pytest.raises(PullError, match="lists thumbs/f03.jpg, which did not arrive"):
        pulling(tmp_path)


def rehash(root: Path, listed: str) -> None:
    """Re-sign one listed file in the manifest after a test has broken it."""
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["files"][listed] = sha(root / listed)
    (root / "manifest.json").write_text(json.dumps(manifest))


def break_role(root: Path) -> None:
    motion_sheet = json.loads((root / "motion.json").read_text())
    del motion_sheet["frames"][0]["role"]
    (root / "motion.json").write_text(json.dumps(motion_sheet))
    rehash(root, "motion.json")


def break_start(root: Path) -> None:
    manifest = json.loads((root / "manifest.json").read_text())
    manifest["clip"]["start"] = "abc"
    (root / "manifest.json").write_text(json.dumps(manifest))


def list_manifest(root: Path) -> None:
    (root / "manifest.json").write_text(json.dumps(["shuffle-3"]))


def list_motion_sheet(root: Path) -> None:
    (root / "motion.json").write_text(json.dumps(["shuffle-3"]))
    rehash(root, "motion.json")


@pytest.mark.parametrize(
    "breaks, said",
    [
        (break_role, "KeyError: 'role'"),
        (break_start, "ValueError"),
        (list_manifest, "holds a JSON list, not an object"),
        (list_motion_sheet, "AttributeError"),
    ],
)
def test_a_malformed_bundle_is_refused_not_a_traceback(tmp_path, monkeypatch, capsys, breaks, said):
    breaks(tracer_bundle(tmp_path / "remote"))
    with pytest.raises(PullError, match=said):
        pulling(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["motions", "remote"]
    assert not (tmp_path / "motions" / "shuffle").exists(), "nothing is installed"
    monkeypatch.setattr(pull.subprocess, "run", fake_rclone(tmp_path / "remote", []))
    monkeypatch.setattr(pull.shutil, "which", which)
    args = ["motions", "pull", "shuffle/shuffle-3", "--motion-root", str(tmp_path / "m"), "--remote", REMOTE]
    assert cli.main(args) == 1
    assert "[pull] REFUSED: " in capsys.readouterr().err


def test_a_list_manifest_elsewhere_in_the_library_is_a_refusal_not_a_traceback(tmp_path):
    tracer_bundle(tmp_path / "remote")
    other = tmp_path / "motions" / "club" / "club-1"
    other.mkdir(parents=True)
    (other / "manifest.json").write_text("[1, 2, 3]")
    with pytest.raises(PullError, match="library\\(\\) does not load .*nothing was installed"):
        pulling(tmp_path)
    assert not (tmp_path / "motions" / "shuffle" / "shuffle-3").exists()


def test_head_boxes_that_do_not_cover_the_clip_are_refused(tmp_path):
    tracer_bundle(tmp_path / "remote", heads=[None] * 119)
    with pytest.raises(PullError, match="holds 119 entries for 120 clip frames"):
        pulling(tmp_path)


def test_a_clip_that_decodes_otherwise_than_declared_is_refused(tmp_path):
    tracer_bundle(tmp_path / "remote")

    def probe(path):
        frames = 118 if path.name == "mask.mp4" else 120
        return SimpleNamespace(size=(720, 1280), frames=frames, variable=False)

    with pytest.raises(PullError, match="mask.mp4 is 118 frames at 720x1280; the clip block says 120"):
        pulling(tmp_path, probe=probe)


def test_the_pull_command_prints_the_install_line(tmp_path, monkeypatch, capsys):
    tracer_bundle(tmp_path / "remote")
    monkeypatch.setattr(pull.subprocess, "run", fake_rclone(tmp_path / "remote", []))
    monkeypatch.setattr(pull.shutil, "which", which)
    probed = []
    monkeypatch.setattr(
        cli.clips, "probe",
        lambda path: probed.append(path.name) or SimpleNamespace(size=(720, 1280), frames=120, variable=False),
    )
    monkeypatch.setenv(pull.REMOTE_ENV, REMOTE)
    root = tmp_path / "motions"
    assert cli.main(["motions", "pull", "shuffle/shuffle-3", "--motion-root", str(root)]) == 0
    out, err = capsys.readouterr()
    assert out.strip() == (
        "shuffle-3: 16f @ 4fps front loop seam='clean' photos=16 airborne=[] travel="
        f"{read_bundle(root / 'shuffle' / 'shuffle-3').load().travel:.3f} clip=ok mask=ok heads=ok"
    )
    assert "kadrive:MotionArtist/shuffle/shuffle-3 ->" in err and "replaced" not in err
    assert probed == ["clip.mp4", "mask.mp4"], "the clip and mask decode as declared"
    assert cli.main(["motions", "pull", "shuffle/shuffle-3", "--motion-root", str(root)]) == 0
    assert "replaced the copy already installed" in capsys.readouterr().err


def test_the_pull_command_refuses_by_name(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pull.subprocess, "run", fake_rclone(tmp_path / "remote", []))
    monkeypatch.setattr(pull.shutil, "which", which)
    args = ["motions", "pull", "shuffle/shuffle-3", "--motion-root", str(tmp_path / "m"), "--remote", REMOTE]
    assert cli.main(args) == 1
    assert "[pull] REFUSED:" in capsys.readouterr().err


def test_list_shows_what_is_on_the_remote_and_what_is_installed(tmp_path, monkeypatch, capsys):
    tracer_bundle(tmp_path / "remote")
    tracer_bundle(tmp_path / "remote", name="shuffle-4")
    tracer_bundle(tmp_path / "remote", name="club-1", set_name="club")
    (tmp_path / "remote" / "club" / "notes").mkdir()
    monkeypatch.setattr(pull.subprocess, "run", fake_rclone(tmp_path / "remote", []))
    monkeypatch.setattr(pull.shutil, "which", which)
    monkeypatch.setenv(pull.REMOTE_ENV, REMOTE)
    root = tmp_path / "motions"
    tracer_bundle(root, name="shuffle-4")
    assert cli.main(["motions", "pull", "--list", "--motion-root", str(root)]) == 0
    lines = [line.split() for line in capsys.readouterr().out.splitlines()]
    assert lines == [["club/club-1", "-"], ["shuffle/shuffle-3", "-"], ["shuffle/shuffle-4", "installed"]]
    assert cli.main(["motions", "pull", "--list", "shuffle", "--motion-root", str(root)]) == 0
    assert [line.split()[0] for line in capsys.readouterr().out.splitlines()] == [
        "shuffle/shuffle-3", "shuffle/shuffle-4"
    ]


def test_motions_with_no_verb_still_lists_the_library(capsys):
    assert cli.main(["motions"]) == 0
    assert "sample" in capsys.readouterr().out
