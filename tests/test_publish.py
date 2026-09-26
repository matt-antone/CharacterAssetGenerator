import fcntl
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


def test_rclone_never_reads_the_builds_terminal(tmp_path):
    """An encrypted config would prompt for its password on a captured stderr and
    wait on the terminal's stdin until the timeout; with no stdin it fails at once."""
    run = Rclone()
    publish.publish(package(tmp_path, "belter"), tmp_path, REMOTE, run, installed)
    [(argv, kwargs)] = run.calls
    assert kwargs["stdin"] is subprocess.DEVNULL
    assert "--ask-password=false" in argv


def test_rclones_own_retries_are_bounded_inside_the_timeout():
    argv = rclone_argv(Path("outputs/belter"), f"{REMOTE}/belter")
    flag = lambda name: argv[argv.index(name) + 1]
    assert flag("--retries") == "1" and flag("--low-level-retries") == "3"
    assert flag("--contimeout") == "15s" and flag("--timeout") == "60s"


def test_a_publish_waits_for_its_turn_only_so_long(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(publish, "LOCK_WAIT", 0.2)
    monkeypatch.setattr(publish, "LOCK_POLL", 0.02)
    folder = package(tmp_path, "belter")
    run = Rclone()
    with open(tmp_path / publish.LOCK, "w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)  # another build's copy, hung
        assert publish.publish(folder, tmp_path, REMOTE, run, installed) is False
    err = capsys.readouterr().err
    assert run.calls == [] and "WARNING" in err and publish.LOCK in err
    assert publish.publish(folder, tmp_path, REMOTE, run, installed) is True  # its turn, once free


@pytest.mark.parametrize("remote", ["gdrive", "outputs/shared"])
def test_a_remote_without_a_colon_is_refused_not_copied_into_a_local_folder(tmp_path, capsys, remote):
    run = Rclone()
    assert publish.publish(package(tmp_path, "belter"), tmp_path, remote, run, installed) is False
    err = capsys.readouterr().err
    assert run.calls == [] and "names no rclone remote" in err and "gdrive:" in err


def test_an_absolute_local_path_is_still_a_destination(tmp_path):
    run = Rclone()
    share = str(tmp_path / "share")
    assert publish.publish(package(tmp_path, "out", "belter"), tmp_path / "out", share, run, installed)
    assert run.calls[0][0][3] == f"{share}/belter"


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
        ("Failed to load config file: unable to decrypt configuration and not allowed to ask "
         "for password - set RCLONE_CONFIG_PASS to your configuration password", "export RCLONE_CONFIG_PASS"),
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


def test_publish_command_goes_on_past_a_brief_it_cannot_read(tmp_path, rclone, capsys):
    out = tmp_path / "out"
    package(out, "velvet-lou")
    broken = tmp_path / "broken.json"
    broken.write_text("{}")
    code = cli.main(["publish", "tests/fixtures/velvet-lou.json", "tests/fixtures/typo.jsn",
                     str(broken), "--out", str(out), "--output-remote", REMOTE])
    assert code == 1
    assert [argv[3] for argv, _ in rclone.calls] == [f"{REMOTE}/velvet-lou"]
    err = capsys.readouterr().err
    assert "typo.jsn" in err and "broken.json" in err


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


def test_an_edit_save_publishes_the_package_it_rewrote(tmp_path, rclone):
    from cag.assemble import tile
    from cag.edit import save
    from tests.test_assemble import cells

    out = tmp_path / "outputs"
    folder = package(out, "default", "belter")
    tile(cells(tmp_path, count=2), folder / "hop-sheet.png")
    png = (folder / "hop-sheet.png").read_bytes()

    assert save(out, "default/belter", "hop-sheet.png", png, 4) == "saved, hop-proof.gif rebuilt"
    assert rclone.calls == []  # no remote, nothing published
    assert save(out, "default/belter", "hop-sheet.png", png, 4, REMOTE).endswith("published")
    assert [argv[3] for argv, _ in rclone.calls] == [f"{REMOTE}/default/belter"]


def test_edit_takes_the_remote_flags(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "serve", lambda out, port, remote=None: seen.update(remote=remote))
    monkeypatch.setenv(publish.REMOTE_ENV, REMOTE)
    assert cli.main(["edit", str(tmp_path)]) == 0 and seen["remote"] == REMOTE
    assert cli.main(["edit", str(tmp_path), "--no-publish"]) == 0 and seen["remote"] is None
