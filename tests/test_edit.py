import json
from io import BytesIO
from pathlib import Path

import pytest
from PIL import Image, ImageSequence

from cag.assemble import gif_proof, tile
from cag.edit import save_frame_sheet
from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from tests.test_assemble import cells


def test_save_overwrites_sheet_and_rebuilds_proof_without_blank_tail(tmp_path):
    paths = cells(tmp_path, count=3)
    sheet = tile(paths, tmp_path / "out" / "hop-sheet.png", columns=2)  # 4th cell blank
    gif_proof(paths, tmp_path / "out" / "hop-proof.gif", fps=12, playback="once")

    with Image.open(sheet) as image:
        edited = image.copy()
    edited.paste((255, 0, 0, 255), (300, 300, 310, 310))
    png = BytesIO()
    edited.save(png, format="PNG")

    proof = save_frame_sheet(tmp_path / "out", "hop-sheet.png", png.getvalue(), fps=10)

    with Image.open(sheet) as image:
        assert image.getpixel((305, 305)) == (255, 0, 0, 255)
    with Image.open(proof) as gif:
        assert sum(1 for _ in ImageSequence.Iterator(gif)) == 3
        assert gif.info["duration"] == 100
        assert gif.info.get("loop", 0) != 0  # build said play once; the edit keeps it


def test_save_rebuilds_a_pingpong_proof_as_a_pingpong(tmp_path):
    """The sheet holds the frames once; how they are played comes off the manifest."""
    from cag.assemble import MANIFEST, manifest

    paths = cells(tmp_path, count=4)
    tile(paths, tmp_path / "hop-sheet.png", columns=4)
    manifest(tmp_path / MANIFEST, "Tall Tom", "6'", {}, [
        {"set_name": "hop", "frames": 4, "fps": 12, "playback": "pingpong", "columns": 4,
         "sheet": "hop-sheet.png", "proof": "hop-proof.gif"}
    ])

    proof = save_frame_sheet(
        tmp_path, "hop-sheet.png", (tmp_path / "hop-sheet.png").read_bytes(), fps=8
    )
    with Image.open(proof) as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 6
        assert gif.info["loop"] == 0


def test_save_refuses_anything_but_an_existing_sheet(tmp_path):
    png = BytesIO()
    Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT)).save(png, format="PNG")
    with pytest.raises(ValueError):
        save_frame_sheet(tmp_path, "../hop-sheet.png", png.getvalue(), fps=12)


def test_server_only_takes_saves_from_its_own_page(tmp_path):
    import threading
    import time
    import urllib.error
    import urllib.request

    from cag.edit import serve

    paths = cells(tmp_path, count=2)
    tile(paths, tmp_path / "hop-sheet.png")
    png = (tmp_path / "hop-sheet.png").read_bytes()
    port = 8799
    threading.Thread(target=serve, args=(tmp_path, port), daemon=True).start()
    time.sleep(0.3)

    def post(origin):
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/save?name=hop-sheet.png&fps=4&folder={tmp_path.name}", data=png, method="POST"
        )
        if origin:
            request.add_header("Origin", origin)
        try:
            return urllib.request.urlopen(request).status
        except urllib.error.HTTPError as error:
            return error.code

    assert post("https://evil.example") == 403
    assert post(None) == 403
    assert post(f"http://127.0.0.1:{port}") == 200


def test_locate_tells_characters_apart_by_contents(tmp_path, monkeypatch):
    from cag.edit import locate

    monkeypatch.chdir(tmp_path)  # find_spec reads specs/ from where cag runs
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "x.json").write_text(
        '{"name": "Tall Tom", "height": "6\'", "description": "d"}'
    )
    for slug, colour in (("tall-tom", (255, 0, 0, 255)), ("belter", (0, 0, 255, 255))):
        (tmp_path / slug).mkdir()
        png = BytesIO()
        Image.new("RGBA", (560, 560), colour).save(png, format="PNG")
        (tmp_path / slug / "dance-sheet.png").write_bytes(png.getvalue())

    tom = (tmp_path / "tall-tom" / "dance-sheet.png").read_bytes()
    assert locate(tmp_path, "dance-sheet.png", tom) == {
        "folder": "tall-tom", "height": "6'", "row": 133, "fps": None  # 528 - 72in at 560px/102in
    }
    belter = (tmp_path / "belter" / "dance-sheet.png").read_bytes()
    assert locate(tmp_path, "dance-sheet.png", belter)["row"] is None  # no brief for belter
    assert locate(tmp_path / "belter", "dance-sheet.png", tom) is None


def test_save_updates_the_manifest_fps(tmp_path):
    from cag.assemble import MANIFEST, manifest

    paths = cells(tmp_path, count=2)
    tile(paths, tmp_path / "hop-sheet.png")
    block = {"set_name": "hop", "frames": 2, "fps": 12, "playback": "loop", "columns": 2,
             "sheet": "hop-sheet.png", "proof": "hop-proof.gif"}
    manifest(tmp_path / MANIFEST, "Tall Tom", "6'", {}, [block])

    save_frame_sheet(tmp_path, "hop-sheet.png", (tmp_path / "hop-sheet.png").read_bytes(), 8)
    written = json.loads((tmp_path / MANIFEST).read_text())
    assert written["sets"]["hop"]["fps"] == 8
    assert written["sets"]["hop"]["frames"] == 2
    assert written["cell"] == [CELL_WIDTH, CELL_HEIGHT]


def test_edit_finds_the_output_tree_from_inside_a_package(tmp_path, monkeypatch):
    from cag.edit import locate, out_root

    package = tmp_path / "outputs" / "halloween" / "tall-tom"
    package.mkdir(parents=True)
    (tmp_path / "specs").mkdir()
    (tmp_path / "specs" / "x.json").write_text(
        '{"name": "Tall Tom", "height": "6\'", "description": "d"}'
    )
    png = BytesIO()
    Image.new("RGBA", (560, 560), (255, 0, 0, 255)).save(png, format="PNG")
    (package / "dance-sheet.png").write_bytes(png.getvalue())

    monkeypatch.chdir(package)  # `cag edit` run from inside one character's folder
    root = out_root(Path("outputs"))  # the default, which is no folder from here
    assert root == (tmp_path / "outputs").resolve()
    found = locate(root, "dance-sheet.png", png.getvalue())
    assert found["folder"] == "halloween/tall-tom"
    assert found["height"] == "6'"  # the roster found beside the tree, not under the cwd
