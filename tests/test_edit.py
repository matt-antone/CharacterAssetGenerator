from io import BytesIO

import pytest
from PIL import Image, ImageSequence

from cag.assemble import gif_proof, sprite_sheet
from cag.edit import save_sheet
from cag.geometry import CELL_HEIGHT, CELL_WIDTH
from tests.test_assemble import cells


def test_save_overwrites_sheet_and_rebuilds_proof_without_blank_tail(tmp_path):
    paths = cells(tmp_path, count=3)
    sheet = sprite_sheet(paths, tmp_path / "out" / "hop-sheet.png", columns=2)  # 4th cell blank
    gif_proof(paths, tmp_path / "out" / "hop-proof.gif", fps=12, loop=False)

    with Image.open(sheet) as image:
        edited = image.copy()
    edited.paste((255, 0, 0, 255), (300, 300, 310, 310))
    png = BytesIO()
    edited.save(png, format="PNG")

    proof = save_sheet(tmp_path / "out", "hop-sheet.png", png.getvalue(), fps=10)

    with Image.open(sheet) as image:
        assert image.getpixel((305, 305)) == (255, 0, 0, 255)
    with Image.open(proof) as gif:
        assert sum(1 for _ in ImageSequence.Iterator(gif)) == 3
        assert gif.info["duration"] == 100
        assert gif.info.get("loop", 0) != 0  # build said play once; the edit keeps it


def test_save_refuses_anything_but_an_existing_sheet(tmp_path):
    png = BytesIO()
    Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT)).save(png, format="PNG")
    with pytest.raises(ValueError):
        save_sheet(tmp_path, "../hop-sheet.png", png.getvalue(), fps=12)
