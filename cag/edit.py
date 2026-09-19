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

from .assemble import gif_proof, split_sheet
from .geometry import ANIM_CONTACT_ROW, CELL_HEIGHT, CELL_WIDTH, anim_subject_height_px
from .spec import load_spec

EDITOR = Path(__file__).with_name("editor.html")


def save_sheet(out_dir: Path | str, name: str, png: bytes, fps: int) -> Path:
    """Overwrite an existing sheet with an edited one and rebuild its proof."""
    dst = Path(out_dir) / Path(name).name
    if not dst.name.endswith("-sheet.png") or not dst.is_file():
        raise ValueError(f"no sprite sheet called {dst.name} in {out_dir}")
    with Image.open(dst) as before:
        size = before.size
    with Image.open(BytesIO(png)) as image:
        if image.size != size:
            raise ValueError(f"edited sheet is {image.size}, the original is {size}")
    columns = size[0] // CELL_WIDTH
    cells = split_sheet(BytesIO(png), columns * (size[1] // CELL_HEIGHT), columns)
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
    return proof


def find_spec(out_dir: Path, specs: Path = Path("specs")) -> Path | None:
    """The brief whose slug names this output folder, if one is in `specs`."""
    for path in sorted(specs.glob("*.json")):
        try:
            if load_spec(path).slug == Path(out_dir).resolve().name:
                return path
        except ValueError:
            continue  # the schema, or a brief that doesn't load
    return None


def folders(root: Path) -> list[Path]:
    """Where sheets may live: the folder `cag edit` was given, and the ones inside it."""
    return [root, *sorted(p for p in root.iterdir() if p.is_dir())]


def locate(root: Path, name: str, png: bytes) -> dict | None:
    """Find which folder an opened sheet came from, by its bytes, and that brief's height.

    The browser hands over a file's name but never its folder, and every set's
    sheet is called `<set>-sheet.png` whatever the character. Only the contents
    tell a heavyweight's dance sheet from a belter's.
    """
    for folder in folders(root):
        path = folder / Path(name).name
        if path.is_file() and path.read_bytes() == png:
            found = {"folder": folder.relative_to(root).as_posix(), "height": None, "row": None}
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
                proof = save_sheet(out_dir / folder, name, body, int(query["fps"][0]))
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
