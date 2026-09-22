"""`cag edit` — serve the frame nudger over one output folder, so Save lands there.

The editor moves pixels; this side only writes the sheet back over itself and
rebuilds the proof beside it, the same way `build` made it.
"""

from __future__ import annotations

import json
import sys
import tempfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from PIL import Image

from .assemble import MANIFEST, gif_proof, split_frame_sheet
from .geometry import ANIM_CONTACT_ROW, CELL_HEIGHT, CELL_WIDTH, anim_subject_height_px
from .spec import load_spec

EDITOR = Path(__file__).with_name("editor.html")


def save_frame_sheet(out_dir: Path | str, name: str, png: bytes, fps: int) -> Path:
    """Overwrite an existing sheet with an edited one and rebuild its proof."""
    dst = Path(out_dir) / Path(name).name
    if not dst.name.endswith("-sheet.png") or not dst.is_file():
        raise ValueError(f"no frame sheet called {dst.name} in {out_dir}")
    with Image.open(dst) as before:
        size = before.size
    with Image.open(BytesIO(png)) as image:
        if image.size != size:
            raise ValueError(f"edited sheet is {image.size}, the original is {size}")
    columns = size[0] // CELL_WIDTH
    cells = split_frame_sheet(BytesIO(png), columns * (size[1] // CELL_HEIGHT), columns)
    while len(cells) > 1 and not cells[-1].getchannel("A").getbbox():
        cells.pop()  # blank tail of the last row

    proof = dst.with_name(dst.name.replace("-sheet.png", "-proof.gif"))
    loop = True
    if proof.exists():  # keep whatever looping the build gave it
        with Image.open(proof) as old:
            loop = old.info.get("loop", 0) == 0

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for index, cell in enumerate(cells):
            paths.append(Path(tmp) / f"{index:03d}.png")
            cell.save(paths[-1])
        dst.write_bytes(png)
        gif_proof(paths, proof, fps, loop)
    update_manifest(dst.parent, dst.name, fps, len(cells))
    return proof


def update_manifest(out_dir: Path, sheet: str, fps: int, frames: int) -> None:
    """Keep the manifest's fps and frame count on what was just written.

    The front end plays from the manifest, so an edit at another rate has to land
    there too or the game keeps the build's speed.
    """
    path = out_dir / MANIFEST
    if not path.is_file():
        return  # a folder from before manifests, or one sheet on its own
    data = json.loads(path.read_text())
    for block in data.get("sets", {}).values():
        if block.get("sheet") == sheet:
            block["fps"] = fps
            block["frames"] = frames
            path.write_text(json.dumps(data, indent=2) + "\n")
            return


def find_spec(out_dir: Path, specs: Path = Path("specs")) -> Path | None:
    """The brief whose slug names this output folder, if one is in `specs`.

    Searched all the way down: the roster is grouped into folders, and a brief
    is found by the slug it carries rather than where it was filed. Run from
    inside a package there is no `specs/` below the cwd, so the roster is looked
    for beside the output tree instead.
    """
    if not specs.is_dir():
        specs = next(
            (p / "specs" for p in Path(out_dir).resolve().parents if (p / "specs").is_dir()),
            specs,
        )
    for path in sorted(specs.rglob("*.json")):
        try:
            if load_spec(path).slug == Path(out_dir).resolve().name:
                return path
        except ValueError:
            continue  # the schema, or a brief that doesn't load
    return None


def out_root(given: Path) -> Path:
    """The output tree to serve, wherever `cag edit` was run from.

    Packages are filed `outputs/<theme>/<slug>`, so the tree to serve is the
    `outputs` folder above: run from inside a package, the default `outputs/`
    is no folder at all, and a package folder passed by hand would otherwise
    hide the rest of the cast.
    """
    path = given.resolve()
    for folder in (path, *path.parents):
        if folder.name == "outputs" and folder.is_dir():
            return folder
    # A tree filed somewhere else: a folder of sheets is served with its siblings.
    return path.parent if any(path.glob("*-sheet.png")) else path


def folders(root: Path) -> list[Path]:
    """Where sheets may live: the folder `cag edit` was given, and every folder under it.

    All the way down, not one level: packages are filed by theme, so a
    character's folder is `outputs/<theme>/<slug>`.
    """
    return [root, *sorted(p for p in root.rglob("*") if p.is_dir())]


def frame_sheet_fps(out_dir: Path, sheet: str) -> int | None:
    """The rate this sheet was rendered for, off the manifest the build wrote."""
    path = out_dir / MANIFEST
    if not path.is_file():
        return None
    for block in json.loads(path.read_text()).get("sets", {}).values():
        if block.get("sheet") == sheet:
            return block.get("fps")
    return None


def locate(root: Path, name: str, png: bytes) -> dict | None:
    """Find which folder an opened sheet came from, by its bytes, and that brief's height.

    The browser hands over a file's name but never its folder, and every set's
    sheet is called `<set>-sheet.png` whatever the character. Only the contents
    tell a heavyweight's dance sheet from a belter's.
    """
    for folder in folders(root):
        path = folder / Path(name).name
        if path.is_file() and path.read_bytes() == png:
            found = {
                "folder": folder.relative_to(root).as_posix(),
                "height": None,
                "row": None,
                "fps": frame_sheet_fps(folder, path.name),
            }
            spec_path = find_spec(folder)
            if spec_path is not None:
                spec = load_spec(spec_path)
                found["height"] = spec.height
                found["row"] = ANIM_CONTACT_ROW + 1 - anim_subject_height_px(spec.height_inches)
            return found
    return None


def serve(out_dir: Path, port: int) -> None:
    # Any page the user visits can POST to localhost, and a rebound DNS name can
    # reach it too. Only our own origin, addressed by a local name, gets in.
    hosts = {f"127.0.0.1:{port}", f"localhost:{port}"}
    out_dir = out_root(out_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.headers.get("Host") not in hosts:
                return self.reply(403, b"forbidden", "text/plain")
            self.reply(200, EDITOR.read_bytes(), "text/html; charset=utf-8")

        def do_POST(self):
            host = self.headers.get("Host")
            if host not in hosts or self.headers.get("Origin") != f"http://{host}":
                return self.reply(403, b"forbidden: not from this editor", "text/plain")
            url = urlparse(self.path)
            query = parse_qs(url.query)
            try:
                body = self.rfile.read(int(self.headers["Content-Length"]))
                name = query["name"][0]
                if url.path == "/locate":
                    found = locate(out_dir, name, body)
                    if found is None:
                        raise ValueError(f"{Path(name).name} is not a sheet under {out_dir}")
                    return self.reply(200, json.dumps(found).encode(), "application/json")
                folder = query.get("folder", ["."])[0]
                if folder not in {f.relative_to(out_dir).as_posix() for f in folders(out_dir)}:
                    raise ValueError(f"no folder {folder!r} under {out_dir}")
                proof = save_frame_sheet(out_dir / folder, name, body, int(query["fps"][0]))
            except (KeyError, ValueError, OSError) as error:
                self.reply(400, str(error).encode(), "text/plain")
            else:
                self.reply(200, f"saved, {proof.name} rebuilt".encode(), "text/plain")

        def reply(self, status, body, kind):
            self.send_response(status)
            self.send_header("Content-Type", kind)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    print(f"editing {out_dir} at http://127.0.0.1:{port}/", file=sys.stderr)
    server.serve_forever()
