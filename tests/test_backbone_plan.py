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
