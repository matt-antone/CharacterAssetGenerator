"""Publishing: copy a finished package to the output remote with rclone.

`CAG_OUTPUT_REMOTE` names an rclone destination — `gdrive:CharacterAssetGenerator/outputs`
— and every build copies its package folder there, at the same path it has under
the output root, so `outputs/default/belter/` lands at `<remote>/default/belter/`
and `index.html` keeps `views/` and the frame sheets beside it.

It is `rclone copy`, never `sync`: nothing on the remote is ever deleted, so a
`--set` build whose gallery lists one set does not take the others' files away.
`--checksum` compares by size and hash, so a rebuild that rewrote a file with the
same bytes uploads nothing.

Publishing never fails a build. A missing rclone, an unknown remote, an expired
token or a dropped network is one warning, naming the command and the fix, and
the package stays on disk for `cag publish` to push later.
"""

from __future__ import annotations

import fcntl
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Callable

#: The environment variable naming the output remote. Unset, nothing is published.
REMOTE_ENV = "CAG_OUTPUT_REMOTE"

#: How long one package's copy may take. A package is about 4 MB; this is the
#: bound on a hung connection, not an expected duration.
TIMEOUT = 900

#: Beside the packages in the output root: parallel builds take turns publishing,
#: because two rclone processes creating the same Drive folder at once can each
#: create one, and Drive allows two folders of one name side by side.
LOCK = ".publish.lock"

INSTALL = "sudo pacman -S rclone (or see https://rclone.org/install/)"


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def destination(remote: str, out_dir: Path, out_root: Path) -> str:
    """Where `out_dir` lands on `remote`: its path under `out_root`, appended.

    `gdrive:` and `gdrive:outputs/` both work; a folder outside the output root
    (never a build's) lands under its own name.
    """
    try:
        relative = out_dir.resolve().relative_to(out_root.resolve()).as_posix()
    except ValueError:
        relative = out_dir.name
    base = remote.rstrip("/")
    if relative in ("", "."):
        return base or remote
    return f"{base}{'' if base.endswith(':') else '/'}{relative}"


def rclone_argv(out_dir: Path, target: str, rclone: str = "rclone") -> list[str]:
    """The one command a publish runs. `copy`, never `sync`: nothing is deleted."""
    return [rclone, "copy", str(out_dir), target, "--checksum", "--stats=0"]


def remote_name(remote: str) -> str | None:
    """The configured remote a destination names — `gdrive` in `gdrive:a/b` — or
    None for a plain local path."""
    head, colon, _ = remote.partition(":")
    if not colon or not head or "/" in head or (len(head) == 1 and head.isalpha()):
        return None
    return head


def diagnose(stderr: str, remote: str) -> str:
    """The fix for a failed copy, read off rclone's own complaint."""
    name = remote_name(remote)
    said = stderr.lower()
    if name and ("didn't find section in config file" in said or "not found in config" in said):
        return (f"remote {name!r} is not configured: "
                f"rclone config create {name} drive scope=drive.file")
    if name and any(word in said for word in ("invalid_grant", "token expired", "oauth2")):
        return f"the {name!r} token needs renewing: rclone config reconnect {name}:"
    if "insufficientfilepermissions" in said or "insufficient permission" in said:
        return ("under scope drive.file rclone sees only what it created itself; "
                "let it create the destination folder rather than making it in Drive by hand")
    lines = [line for line in stderr.strip().splitlines() if line.strip()]
    return lines[-1] if lines else "rclone gave no reason"


def publish(
    out_dir: Path,
    out_root: Path,
    remote: str | None,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> bool:
    """Copy one package folder to `remote`. True if it was published.

    With no remote, nothing happens and rclone is never looked for. Every
    failure is logged as a warning and returns False; none raises.
    """
    if not remote:
        return False
    if not out_dir.is_dir():
        log(f"[publish] nothing at {out_dir} to publish")
        return False
    run, which = run or subprocess.run, which or shutil.which
    rclone = which("rclone")
    if not rclone:
        log(f"[publish] WARNING: rclone is not installed, so {out_dir} was not published. "
            f"Install it: {INSTALL}; then uv run cag publish --all")
        return False
    target = destination(remote, out_dir, out_root)
    argv = rclone_argv(out_dir, target, rclone)
    command = " ".join(["rclone", *argv[1:]])
    try:
        out_root.mkdir(parents=True, exist_ok=True)
        with open(out_root / LOCK, "w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            done = run(argv, capture_output=True, text=True, timeout=TIMEOUT)
    except subprocess.TimeoutExpired:
        log(f"[publish] WARNING: `{command}` gave no answer in {TIMEOUT}s; "
            "check the network, then uv run cag publish on this brief")
        return False
    except OSError as error:
        log(f"[publish] WARNING: `{command}` could not run: {error}")
        return False
    if done.returncode != 0:
        log(f"[publish] WARNING: `{command}` failed (exit {done.returncode}): "
            f"{diagnose(done.stderr or '', remote)}. The package is on disk; "
            "uv run cag publish pushes it once that is fixed")
        return False
    log(f"[publish] {out_dir} -> {target}")
    return True


def packages(out_root: Path, manifest: str) -> list[Path]:
    """Every package folder under `out_root`: each folder holding a package manifest."""
    return sorted(path.parent for path in out_root.rglob(manifest) if path.is_file())
