"""SAM 3 pieces that need no model or board: phrase splitting, stretched-image geometry, CLIP BPE."""
import numpy as np

from app.backends import Sam3Backend
from dino.letterbox import Squash


def test_phrases():
    assert Sam3Backend.phrases("cat . remote control.  Cat . red   car,") == ["cat", "remote control", "red car"]
    assert len(Sam3Backend.phrases(".".join("abcdefghij"))) == Sam3Backend.MAX_PHRASES
    assert Sam3Backend.phrases(" . . ") == []


def test_squash_boxes_cover_the_whole_image():
    g = Squash(672, 1280, 720)
    b = g.boxes_to_source(np.array([[0.5, 0.5, 1.0, 1.0], [0.25, 0.5, 0.5, 0.2]], np.float32))
    assert np.allclose(b[0], [0, 0, 1280, 720])
    assert np.allclose(b[1], [0, 288, 640, 432])
    assert g.new_w == g.new_h == 672


def test_clip_bpe_basics(tmp_path):
    from dino.clip_bpe import ClipTokenizer, bytes_to_unicode
    assert len(bytes_to_unicode()) == 256
    (tmp_path / "vocab.json").write_text('{"<|startoftext|>": 0, "<|endoftext|>": 1, "c": 2, "a": 3, "t</w>": 4, '
                                         '"ca": 5, "cat</w>": 6, "!</w>": 7}')
    (tmp_path / "merges.txt").write_text("#version: 0.2\nc a\nca t</w>\n")
    t = ClipTokenizer(tmp_path / "vocab.json", tmp_path / "merges.txt")
    assert t.encode("  CAT ") == [0, 6, 1]
    assert t.encode("cat !") == [0, 6, 7, 1]
    assert len(t.encode("cat " * 40, max_tokens=32)) == 32
