"""Post-processing: candidate boxes, phrase assignment, letterbox geometry."""
import numpy as np

from dino import postprocess as PP
from dino import prompt as P
from dino.letterbox import Letterbox, letterbox


def test_letterbox_geometry():
    from PIL import Image
    img = Image.new("RGB", (640, 480), (124, 116, 104))
    x, lb = letterbox(img, 576)
    assert x.shape == (3, 576, 576)
    assert (lb.new_w, lb.new_h) == (576, 432)
    assert np.abs(x[:, 432:, :]).max() == 0          # padding is 0 after normalisation
    box = lb.boxes_to_source(np.array([[0.5, 0.375, 1.0, 0.75]]))
    assert np.allclose(box, [[0, 0, 640, 480]], atol=0.6)


def test_candidates_and_phrases():
    tok = P.Tokenizer()
    pr = P.build("cat . red car .", tok)
    logits = np.full((900, 64), -10.0, np.float32)
    logits[:, ~pr.token_mask] = -np.inf
    cat_tok = np.nonzero(pr.phrase_of_token == 0)[0]
    car_tok = np.nonzero(pr.phrase_of_token == 1)[0]
    logits[0, cat_tok] = 2.0
    logits[1, car_tok] = 0.0
    boxes = np.tile(np.array([[0.25, 0.25, 0.1, 0.1]], np.float32), (900, 1))
    lb = Letterbox(576, 0.9, 640, 480, 576, 432)
    c = PP.candidates(logits, boxes, pr, lb, floor=0.3)
    assert [d["phrase"] for d in c] == [0, 1]
    assert abs(c[0]["score"] - 1 / (1 + np.exp(-2))) < 1e-3
    assert c[0]["box"][0] < c[0]["box"][2]


def test_nms_per_phrase():
    d = [{"box": [0, 0, 10, 10], "phrase": 0, "score": .9}, {"box": [1, 1, 10, 10], "phrase": 0, "score": .8},
         {"box": [1, 1, 10, 10], "phrase": 1, "score": .7}]
    assert len(PP.nms(d, 0.5)) == 2


def test_nms_np_matches_nms():
    import numpy as np
    from dino import postprocess as PP
    rng = np.random.default_rng(0)
    xy = rng.uniform(0, 500, (200, 2))
    b = np.concatenate([xy, xy + rng.uniform(20, 200, (200, 2))], 1)
    s, lab = rng.uniform(0, 1, 200), rng.integers(0, 3, 200)
    keep = PP.nms_np(b, s, lab, 0.5)
    ref = PP.nms([{"box": list(b[i]), "score": s[i], "phrase": int(lab[i])} for i in np.argsort(-s)], 0.5)
    assert [round(d["score"], 9) for d in ref] == [round(float(s[i]), 9) for i in keep]
