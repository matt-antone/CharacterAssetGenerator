import subprocess
from pathlib import Path

import pytest

from cag import cli, publish
from cag.assemble import MANIFEST
from cag.publish import destination, diagnose, packages, rclone_argv, remote_name

REMOTE = "gdrive:CharacterAssetGenerator/outputs"


class Rclone:
    """A stand-in for `subprocess.run`: records each argv, answers as told."""

    def __init__(self, returncode=0, stderr="", raises=None):
        self.calls, self.returncode, self.stderr, self.raises = [], returncode, stderr, raises

    def __call__(self, argv, **kwargs):
        self.calls.append((argv, kwargs))
        if self.raises:
            raise self.raises
        return subprocess.CompletedProcess(argv, self.returncode, "", self.stderr)


def installed(name):
    return f"/usr/bin/{name}"


def package(root: Path, *parts: str) -> Path:
    folder = root.joinpath(*parts)
    (folder / "views").mkdir(parents=True)
    (folder / MANIFEST).write_text("{}")
    (folder / "index.html").write_text("<img src='views/key.png'>")
    (folder / "views" / "key.png").write_bytes(b"png")
    return folder


def test_the_package_lands_at_its_own_path_under_the_remote(tmp_path):
    out = tmp_path / "outputs"
    assert destination(REMOTE, out / "default" / "belter", out) == f"{REMOTE}/default/belter"
    assert destination(REMOTE + "/", out / "belter", out) == f"{REMOTE}/belter"
    assert destination("gdrive:", out / "default" / "belter", out) == "gdrive:default/belter"


def test_a_folder_outside_the_output_root_lands_under_its_own_name(tmp_path):
    assert destination(REMOTE, tmp_path / "elsewhere" / "mort", tmp_path / "outputs") == f"{REMOTE}/mort"


def test_the_command_copies_and_never_syncs():
    argv = rclone_argv(Path("outputs/default/belter"), f"{REMOTE}/default/belter")
    assert argv[:4] == ["rclone", "copy", "outputs/default/belter", f"{REMOTE}/default/belter"]
    assert "--checksum" in argv
    assert not {"sync", "move", "delete", "--progress", "-P"} & set(argv)


def test_the_remote_name_is_the_part_before_the_colon():
    assert remote_name(REMOTE) == "gdrive"
    assert remote_name("/mnt/share/outputs") is None
    assert remote_name("C:/outputs") is None


def test_no_remote_publishes_nothing_and_never_looks_for_rclone(tmp_path):
    def which(name):
        raise AssertionError("rclone was looked for")

    run = Rclone()
    assert publish.publish(package(tmp_path, "belter"), tmp_path, None, run, which) is False
    assert run.calls == []


def test_a_publish_copies_the_whole_folder(tmp_path):
    folder = package(tmp_path / "outputs", "default", "belter")
    run = Rclone()
    assert publish.publish(folder, tmp_path / "outputs", REMOTE, run, installed) is True
    [(argv, kwargs)] = run.calls
    assert argv[:4] == ["/usr/bin/rclone", "copy", str(folder), f"{REMOTE}/default/belter"]
    assert kwargs["capture_output"] and kwargs["timeout"]


def test_a_folder_that_is_not_there_publishes_nothing(tmp_path, capsys):
    run = Rclone()
    assert publish.publish(tmp_path / "belter", tmp_path, REMOTE, run, installed) is False
    assert run.calls == [] and "nothing at" in capsys.readouterr().err


def test_no_rclone_is_a_warning_with_the_install_command(tmp_path, capsys):
    run = Rclone()
    assert publish.publish(package(tmp_path, "belter"), tmp_path, REMOTE, run, lambda n: None) is False
    err = capsys.readouterr().err
    assert run.calls == [] and "WARNING" in err and "sudo pacman -S rclone" in err


@pytest.mark.parametrize(
    "stderr, fix",
    [
        ('Failed to create file system for "gdrive:x": didn\'t find section in config file',
         "rclone config create gdrive drive scope=drive.file"),
        ("oauth2: cannot fetch token: 400 Bad Request invalid_grant", "rclone config reconnect gdrive:"),
        ("googleapi: Error 403: insufficientFilePermissions", "scope drive.file"),
        ("ERROR : some other trouble\n", "some other trouble"),
    ],
)
def test_a_failed_copy_names_the_command_and_the_fix(tmp_path, capsys, stderr, fix):
    run = Rclone(returncode=1, stderr=stderr)
    assert publish.publish(package(tmp_path, "belter"), tmp_path, REMOTE, run, installed) is False
    err = capsys.readouterr().err
    assert f"rclone copy {tmp_path / 'belter'} {REMOTE}/belter" in err and fix in err


@pytest.mark.parametrize(
    "error", [subprocess.TimeoutExpired("rclone", 1), FileNotFoundError("rclone")]
)
def test_a_hung_or_vanished_rclone_is_a_warning(tmp_path, capsys, error):
    run = Rclone(raises=error)
    assert publish.publish(package(tmp_path, "belter"), tmp_path, REMOTE, run, installed) is False
    assert "WARNING" in capsys.readouterr().err


def test_diagnose_without_a_named_remote_falls_back_to_rclones_words():
    assert diagnose("didn't find section in config file", "/mnt/share") == (
        "didn't find section in config file"
    )
    assert diagnose("", REMOTE) == "rclone gave no reason"


def test_packages_are_the_folders_holding_a_manifest(tmp_path):
    a = package(tmp_path, "default", "belter")
    b = package(tmp_path, "loose")
    (tmp_path / "default" / "half-built").mkdir()
    assert packages(tmp_path, MANIFEST) == [a, b]


@pytest.fixture
def rclone(monkeypatch):
    run = Rclone()
    monkeypatch.setattr(publish.subprocess, "run", run)
    monkeypatch.setattr(publish.shutil, "which", installed)
    return run


def test_publish_command_copies_named_briefs_and_says_which_were_never_built(tmp_path, rclone, capsys):
    out = tmp_path / "out"
    package(out, "velvet-lou")
    code = cli.main(["publish", "tests/fixtures/velvet-lou.json", "tests/fixtures/no-animations.json",
                     "--out", str(out), "--output-remote", REMOTE])
    assert code == 1  # one brief has no package
    assert [argv[3] for argv, _ in rclone.calls] == [f"{REMOTE}/velvet-lou"]
    assert "build it first" in capsys.readouterr().err


def test_publish_all_copies_every_package_and_reads_the_env(tmp_path, rclone, monkeypatch):
    out = tmp_path / "out"
    package(out, "default", "belter")
    package(out, "halloween", "mort")
    monkeypatch.setenv(publish.REMOTE_ENV, REMOTE)
    assert cli.main(["publish", "--all", "--out", str(out)]) == 0
    assert [argv[3] for argv, _ in rclone.calls] == [
        f"{REMOTE}/default/belter", f"{REMOTE}/halloween/mort"
    ]


def test_publish_without_a_remote_says_how_to_name_one(tmp_path, rclone, capsys):
    assert cli.main(["publish", "--all", "--out", str(tmp_path)]) == 1
    assert rclone.calls == [] and publish.REMOTE_ENV in capsys.readouterr().err


def test_a_failed_publish_fails_the_publish_command(tmp_path, monkeypatch):
    package(tmp_path, "belter")
    monkeypatch.setattr(publish.subprocess, "run", Rclone(returncode=1))
    monkeypatch.setattr(publish.shutil, "which", installed)
    assert cli.main(["publish", "--all", "--out", str(tmp_path), "--output-remote", REMOTE]) == 1


BUILD = ["build", "tests/fixtures/velvet-lou.json", "--draw-backend", "codex"]


def test_the_build_flag_overrides_the_env_and_no_publish_turns_it_off(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "build", lambda *a, **kw: seen.update(kw))
    monkeypatch.setenv(publish.REMOTE_ENV, REMOTE)
    args = [*BUILD, "--work", str(tmp_path / "w"), "--out", str(tmp_path / "o")]
    assert cli.main(args) == 0 and seen["remote"] == REMOTE
    assert cli.main([*args, "--output-remote", "other:x"]) == 0 and seen["remote"] == "other:x"
    assert cli.main([*args, "--no-publish"]) == 0 and seen["remote"] is None
    monkeypatch.delenv(publish.REMOTE_ENV)
    assert cli.main(args) == 0 and seen["remote"] is None
