import numpy
import pytest
from PIL import Image, ImageSequence

from cag.assemble import PORTRAIT_SIZES, PROOF_BACKDROP, gif_proof, portrait, split_frame_sheet, tile
from cag.geometry import CELL_HEIGHT, CELL_WIDTH


def cells(tmp_path, count=4):
    paths = []
    for index in range(count):
        image = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
        image.paste((index * 20 + 10, 40, 200, 255), (10, 10, 100, 100))
        path = tmp_path / f"{index:02d}.png"
        image.save(path)
        paths.append(path)
    return paths


def test_sheet_is_a_strip_of_cells_in_order(tmp_path):
    sheet = tile(cells(tmp_path), tmp_path / "sheet.png")
    with Image.open(sheet) as image:
        assert image.size == (CELL_WIDTH * 4, CELL_HEIGHT)
        assert image.getpixel((CELL_WIDTH * 2 + 50, 50))[0] == 50  # third cell


def test_sheet_survives_a_split_round_trip(tmp_path):
    paths = cells(tmp_path)
    sheet = tile(paths, tmp_path / "sheet.png")
    for original, restored in zip(paths, split_frame_sheet(sheet, len(paths))):
        with Image.open(original) as before:
            assert numpy.array_equal(numpy.array(before.convert("RGBA")), numpy.array(restored))


def test_sheet_wraps_onto_rows(tmp_path):
    sheet = tile(cells(tmp_path), tmp_path / "sheet.png", columns=2)
    with Image.open(sheet) as image:
        assert image.size == (CELL_WIDTH * 2, CELL_HEIGHT * 2)
    restored = split_frame_sheet(sheet, 4, columns=2)
    assert restored[3].getpixel((50, 50))[0] == 70  # fourth cell, second row


def test_sheet_rejects_an_unregistered_cell(tmp_path):
    odd = tmp_path / "odd.png"
    Image.new("RGBA", (10, 10)).save(odd)
    with pytest.raises(ValueError, match=f"not the {CELL_WIDTH}x{CELL_HEIGHT} cell"):
        tile([odd], tmp_path / "sheet.png")


def test_proof_loops_at_the_declared_rate(tmp_path):
    proof = gif_proof(cells(tmp_path), tmp_path / "proof.gif", fps=4)
    with Image.open(proof) as gif:
        frames = list(ImageSequence.Iterator(gif))
        assert len(frames) == 4
        assert frames[0].info["duration"] == 250
        assert gif.info["loop"] == 0


def test_a_pingpong_proof_bounces_back_down_its_frames(tmp_path):
    proof = gif_proof(cells(tmp_path), tmp_path / "proof.gif", fps=4, playback="pingpong")
    with Image.open(proof) as gif:
        # Four down and two back up: the ends are played once, not twice.
        assert len(list(ImageSequence.Iterator(gif))) == 6
        assert gif.info["loop"] == 0


def test_a_one_shot_proof_plays_once(tmp_path):
    proof = gif_proof(cells(tmp_path), tmp_path / "proof.gif", fps=4, playback="once")
    with Image.open(proof) as gif:
        assert len(list(ImageSequence.Iterator(gif))) == 4
        assert gif.info.get("loop", 0) != 0


def test_proof_flattens_transparency_onto_grey(tmp_path):
    proof = gif_proof(cells(tmp_path), tmp_path / "proof.gif", fps=4)
    with Image.open(proof) as gif:
        assert gif.convert("RGB").getpixel((400, 400)) == PROOF_BACKDROP


def test_empty_input_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="no cells"):
        tile([], tmp_path / "sheet.png")
    with pytest.raises(ValueError, match="no cells"):
        gif_proof([], tmp_path / "proof.gif", fps=4)


def test_portrait_is_a_square_cut_around_the_head(tmp_path):
    """A figure 400px tall with its head off to one side: the portrait is square,
    starts at the crown and is centred on the head, not the body."""
    cell = Image.new("RGBA", (CELL_WIDTH, CELL_HEIGHT), (0, 0, 0, 0))
    cell.paste((200, 0, 0, 255), (200, 200, 300, 550))  # body
    cell.paste((0, 0, 200, 255), (300, 150, 340, 200))  # head, right of the body
    cell.save(key := tmp_path / "key.png")
    for size in PORTRAIT_SIZES.values():
        out = Image.open(portrait(key, tmp_path / f"portrait-{size}.png", size))
        assert out.size == (size, size)
        box = out.getchannel("A").getbbox()
        assert box[1] <= size // 15  # crown near the top edge
        assert out.getpixel((size // 2, box[1]))[2] == 200  # head sits in the middle column
