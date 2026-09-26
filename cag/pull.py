"""Pull a motion bundle from MotionArtist's Drive into `motions/`.

MotionArtist syncs every bundle it exports to `$MOTION_ARTIST_REMOTE/<set>/<name>/`
as a plain directory, where the name is `<set>-<index>` (`shuffle/shuffle-3`,
the index not zero-padded). The bundle carries its source clip, cut from the
file the tracer traced, and may carry a bundle mask and head boxes beside it;
their hashes are in the manifest's `clip` block.

`pull` copies one such directory into `motions/<set>/<name>/`, unrenamed, and
installs it only once it has checked it: the directory, the manifest's `name`
and `motion.json`'s `name` are one name, every file the manifest lists and
every clip file it declares is there with its hash, the motion sheet loads
against its manifest, and `library()` finds it. The copy lands in a staging
directory first, so a pull that is refused, fails or is interrupted leaves
nothing half-installed. A bundle already installed under the name is replaced:
the tracer re-syncs a bundle in place when it re-exports it.

rclone runs the way `cag.publish` runs it: no stdin, `--ask-password=false`,
bounded retries and a bounded timeout.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .motion import (
    BUNDLE,
    PARTS,
    Bundle,
    MotionError,
    clip_status,
    library,
    part_status,
    read_bundle,
    read_heads,
    sha256_of,
)
from .publish import INSTALL, RCLONE_FLAGS, call, diagnose, names_a_destination

#: The environment variable naming MotionArtist's remote, e.g. `kadrive:MotionArtist`.
REMOTE_ENV = "MOTION_ARTIST_REMOTE"

#: How long one bundle's copy, or one listing, may take. A bundle is its clip,
#: its mask and a few dozen small files; this bounds a hung connection.
TIMEOUT = 600

#: `<set>/<set>-<index>`: the set's own name, then the index, not zero-padded.
TARGET = re.compile(r"^(?P<set>[A-Za-z0-9][A-Za-z0-9_-]*)/(?P<name>(?P=set)-(?:0|[1-9][0-9]*))$")

#: Where a pull stages its copy, beside the motion root so the move into place
#: is a rename. Removed when the pull ends, whatever happened.
STAGING = ".cag-pull-"

#: A 64-character hex digest: a `files` entry that is a hash to check.
_SHA = re.compile(r"^[0-9a-f]{64}$")


class PullError(RuntimeError):
    """Raised when a bundle cannot be pulled, or would not install cleanly."""


@dataclass(frozen=True)
class Pulled:
    """What `pull` installed."""

    bundle: Bundle
    #: Where it came from on the remote.
    source: str
    #: True when a bundle of that name was installed before and was replaced.
    replaced: bool


def parse_target(target: str) -> tuple[str, str]:
    """`shuffle/shuffle-3` as `("shuffle", "shuffle-3")`, or PullError naming the rule."""
    found = TARGET.match(target.strip().strip("/"))
    if not found:
        raise PullError(
            f"{target!r} is not <set>/<set>-<index>: a bundle is named after its set and "
            "an index with no leading zeros, e.g. shuffle/shuffle-3"
        )
    return found["set"], found["name"]


def remote_path(remote: str, *parts: str) -> str:
    """`parts` joined under `remote`, whether it ends in a colon, a slash or neither."""
    base = remote.rstrip("/")
    tail = "/".join(parts)
    if not tail:
        return base or remote
    return f"{base}{'' if base.endswith(':') else '/'}{tail}"


def _rclone(
    argv: list[str],
    remote: str,
    run: Callable[..., subprocess.CompletedProcess],
) -> subprocess.CompletedProcess:
    command = " ".join(["rclone", *argv[1:]])
    try:
        done = call(run, argv, TIMEOUT)
    except subprocess.TimeoutExpired:
        raise PullError(f"`{command}` gave no answer in {TIMEOUT}s; check the network") from None
    except OSError as error:
        raise PullError(f"`{command}` could not run: {error}") from None
    if done.returncode != 0:
        said = done.stderr or ""
        fix = (
            "there is nothing at that path on the remote; `cag motions pull --list` shows "
            "what is there"
            if "directory not found" in said.lower()
            else diagnose(said, remote)
        )
        raise PullError(f"`{command}` failed (exit {done.returncode}): {fix}")
    return done


def _rclone_path(remote: str | None, which: Callable[[str], str | None]) -> str:
    if not remote:
        raise PullError(
            f"no remote: set {REMOTE_ENV} (e.g. kadrive:MotionArtist) or pass --remote"
        )
    if not names_a_destination(remote):
        raise PullError(
            f"{remote!r} names no rclone remote; put a colon after the remote's name, "
            "e.g. kadrive:MotionArtist"
        )
    rclone = which("rclone")
    if not rclone:
        raise PullError(f"rclone is not installed. Install it: {INSTALL}")
    return rclone


def list_remote(
    remote: str | None,
    set_name: str = "",
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    which: Callable[[str], str | None] | None = None,
) -> list[str]:
    """Every bundle on the remote, or in one set of it, as `<set>/<name>`, sorted."""
    run, which = run or subprocess.run, which or shutil.which
    rclone = _rclone_path(remote, which)
    if set_name:
        argv = [rclone, "lsf", "--dirs-only", remote_path(remote, set_name), *RCLONE_FLAGS]
    else:
        argv = [rclone, "lsf", "-R", "--dirs-only", "--max-depth", "2", remote, *RCLONE_FLAGS]
    listed = _rclone(argv, remote, run).stdout.splitlines()
    found = []
    for line in listed:
        entry = line.strip().strip("/")
        entry = f"{set_name}/{entry}" if set_name else entry
        if TARGET.match(entry):
            found.append(entry)
    return sorted(found)


#: What a malformed manifest or motion sheet raises on its way through
#: `read_bundle` and `load_motion`: a missing key, a field that is not a number,
#: a JSON list where an object belongs. Each is one refused pull, never a traceback.
MALFORMED = (MotionError, KeyError, IndexError, ValueError, TypeError, AttributeError)


def check(
    root: Path,
    name: str,
    probe: Callable[[Path], object] | None = None,
) -> Bundle:
    """The bundle at `root`, once it has passed every check a pull makes; else PullError.

    `probe` is `cag.clips.probe`, to check the clip and bundle mask decode at
    the frame count and size the manifest declares; None skips that. Sizes are
    display sizes, the way a drive build decodes the clip, so a clip stored on
    its side with a rotation matrix is checked upright.
    """
    try:
        return _check(root, name, probe)
    except PullError:
        raise
    except MALFORMED as error:
        said = error if isinstance(error, MotionError) else f"{type(error).__name__}: {error}"
        raise PullError(f"{name} does not read as a bundle: {said}") from None


def _check(root: Path, name: str, probe: Callable[[Path], object] | None) -> Bundle:
    manifest = root / BUNDLE
    if not manifest.exists():
        raise PullError(f"{name} holds no {BUNDLE}: nothing is there, or it is not a bundle")
    try:
        data = json.loads(manifest.read_text())
    except json.JSONDecodeError as error:
        raise PullError(f"{name}'s {BUNDLE} is not valid JSON: {error}") from None
    if not isinstance(data, dict):
        raise PullError(f"{name}'s {BUNDLE} holds a JSON {type(data).__name__}, not an object")
    if data.get("name") != name:
        raise PullError(
            f"{name}'s manifest calls it {data.get('name')!r}: the directory, the manifest's "
            "name and motion.json's name must be one name"
        )
    files = data.get("files") or {}
    for listed in files:
        path = root / listed
        if not path.is_file():
            raise PullError(f"{name}'s manifest lists {listed}, which did not arrive")
        want = files[listed] if isinstance(files, dict) else ""
        if isinstance(want, str) and _SHA.match(want) and sha256_of(path) != want:
            raise PullError(f"{name}'s {listed} is not the file its manifest hashes")
    motion_files = [f for f in files if Path(f).name == "motion.json"]
    if len(motion_files) == 1:
        try:
            said = json.loads((root / motion_files[0]).read_text()).get("name")
        except json.JSONDecodeError as error:
            raise PullError(f"{name}'s {motion_files[0]} is not valid JSON: {error}") from None
        if said != name:
            raise PullError(
                f"{name}'s motion.json calls it {said!r}: the directory, the manifest's name "
                "and motion.json's name must be one name"
            )
    bundle = read_bundle(root)
    motion = bundle.load()
    if not bundle.photos:
        raise PullError(
            f"{name} ships {len([f for f in files if Path(f).parent.name == 'thumbs'])} traced "
            f"frames for {bundle.frame_count} frames: the set would render with no pose reference"
        )
    declared = bundle.declared_clip
    if declared:
        status = clip_status(bundle)
        if status != "ok":
            raise PullError(f"{name}'s clip {declared.path.name} is {status}")
        for kind in PARTS:
            part = getattr(declared, kind)
            if part is None:
                continue
            status = part_status(bundle, kind)
            if status != "ok":
                raise PullError(f"{name}'s {kind} {part.path.name} is {status}")
        if declared.heads:
            try:
                read_heads(declared.heads.path, declared.frame_count)
            except MotionError as error:
                raise PullError(f"{name}: {error}") from None
        if probe is not None:
            _check_video(name, declared.path, declared.frame_count, declared.size, probe)
            if declared.mask:
                _check_video(name, declared.mask.path, declared.frame_count, declared.size, probe)
    elif motion.clip_frames:
        raise PullError(f"{name}'s motion.json counts clip frames, but it ships no clip")
    return bundle


def _check_video(
    name: str, path: Path, frames: int, size: tuple[int, int], probe: Callable[[Path], object]
) -> None:
    try:
        found = probe(path)
    except Exception as error:  # noqa: BLE001 - one probe failure is one refusal
        raise PullError(f"{name}'s {path.name} does not probe: {error}") from None
    if tuple(found.size) != tuple(size) or found.frames != frames:
        raise PullError(
            f"{name}'s {path.name} is {found.frames} frames at {found.size[0]}x{found.size[1]}; "
            f"the clip block says {frames} at {size[0]}x{size[1]}"
        )
    if getattr(found, "variable", False):
        raise PullError(f"{name}'s {path.name} has a variable frame rate")


def _elsewhere(motion_root: Path, name: str, home: Path) -> Path | None:
    """Another bundle directory under `motion_root` whose manifest takes `name`."""
    for manifest in sorted(motion_root.rglob(BUNDLE)):
        if manifest.parent.resolve() == home.resolve():
            continue
        try:
            data = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        # Someone else's broken manifest is not this pull's to refuse.
        called = data.get("name", manifest.parent.name) if isinstance(data, dict) else None
        if called == name:
            return manifest.parent
    return None


def pull(
    target: str,
    motion_root: Path,
    remote: str | None,
    run: Callable[..., subprocess.CompletedProcess] | None = None,
    which: Callable[[str], str | None] | None = None,
    probe: Callable[[Path], object] | None = None,
) -> Pulled:
    """Copy `<set>/<name>` from `remote` into `motion_root/<set>/<name>/`, checked.

    Raises PullError, with nothing installed and anything installed before
    left as it was, when the copy fails or the bundle fails a check.
    """
    set_name, name = parse_target(target)
    run, which = run or subprocess.run, which or shutil.which
    rclone = _rclone_path(remote, which)
    motion_root = Path(motion_root)
    home = motion_root / set_name / name
    other = _elsewhere(motion_root, name, home)
    if other is not None:
        raise PullError(
            f"{other} already calls itself {name}; two bundles of one name and a brief "
            "gets whichever library() reads last"
        )
    source = remote_path(remote, set_name, name)
    motion_root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=STAGING, dir=motion_root.resolve().parent))
    try:
        staged = staging / name
        _rclone(
            [rclone, "copy", source, str(staged), "--checksum", "--stats=0", *RCLONE_FLAGS],
            remote,
            run,
        )
        check(staged, name, probe)
        replaced = home.exists()
        previous = staging / "previous"
        home.parent.mkdir(parents=True, exist_ok=True)
        if replaced:
            home.rename(previous)
        staged.rename(home)
        try:
            bundle = library(motion_root)[name]
            bundle.load()
        except (*MALFORMED, OSError) as error:
            shutil.rmtree(home, ignore_errors=True)
            if replaced:
                previous.rename(home)
            raise PullError(
                f"with {name} installed, library() does not load ({error}); "
                + ("the copy that was there is back" if replaced else "nothing was installed")
            ) from None
        return Pulled(bundle, source, replaced)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
