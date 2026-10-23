"""The backbone program's gather tables are valid permutations / merges."""
import re

import numpy as np
import pytest

import backbone_plan as BP


@pytest.mark.parametrize("Hp,shift", [(144, 0), (144, 1), (36, 1), (24, 0), (24, 1)])
def test_layout_is_permutation(Hp, shift):
    order = BP.layout(Hp, Hp, shift)
    assert sorted(order) == list(range(Hp * Hp))


def test_shifted_layout_matches_roll():
    Hp = 24
    order = BP.layout(Hp, Hp, 1)
    # first window, first token is spatial (6, 6) after rolling by -6
    assert order[0] == 6 * Hp + 6


def test_plan_tables():
    from transformers import AutoModelForZeroShotObjectDetection
    hf = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-base").eval()
    steps = BP.build(hf, 576)
    for st in steps:
        if st.op == "gather" and st.k == 1 and re.match(r"s\d_b\d+_to_b", st.name):
            assert sorted(st.table) == list(range(len(st.table))), st.name
        if st.op == "gather" and st.k == 4:
            t = st.table[st.table >= 0]
            assert len(set(t.tolist())) == len(t), st.name
    assert sum(s.op == "run" for s in steps) == 27


@pytest.mark.parametrize("ws,shift,Hp", [(12, 6, 24), (12, 6, 36), (7, 3, 21), (7, 3, 147), (7, 3, 14)])
def test_shift_by_slices_matches_roll(ws, shift, Hp):
    """Swin-B (12/6) and Swin-T (7/3) window shifts with Slice/Concat equal roll + partition."""
    import torch
    import dino_mla as M
    import swin_shift_mla as SS
    n = Hp // ws
    x = torch.randn(1, 3, Hp, Hp)
    plain = M.window_partition(x, ws)
    shifted = SS.plain_to_shifted(plain, n, n, ws, shift)
    assert torch.equal(shifted, M.window_partition(M.roll_hw(x, -shift), ws))
    assert torch.equal(SS.shifted_to_plain(shifted, n, n, ws, shift), plain)


@pytest.mark.parametrize("ws,shift,Hp", [(7, 3, 147), (7, 3, 21)])
def test_tiny_layouts_are_permutations(ws, shift, Hp):
    for sh in (0, 1):
        assert sorted(BP.layout(Hp, Hp, sh, ws, shift)) == list(range(Hp * Hp))
