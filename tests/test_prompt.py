"""Prompt handling matches transformers' Grounding DINO processing."""
import numpy as np
import pytest
import torch

from dino import prompt as P

PROMPTS = ["a cat. a remote control.", "person . red car . traffic light .", "Dog", "mug, laptop; phone\nkeyboard",
           "the man's hat . 3D printer . x-ray .", "naïve café .", "a . b . c . d . e . f . g . h ."]


@pytest.fixture(scope="module")
def tok():
    return P.Tokenizer()


def test_normalize():
    assert P.normalize("Cat, DOG;  red   car\n") == "cat . dog . red car ."
    assert P.normalize("  ") == ""


@pytest.mark.parametrize("text", PROMPTS)
def test_masks_match_hf(tok, text):
    from transformers.models.grounding_dino.modeling_grounding_dino import \
        generate_masks_with_special_tokens_and_transfer_map
    pr = P.build(text, tok)
    n = pr.n_tokens
    ids = torch.from_numpy(pr.input_ids[:n])[None]
    m, pos = generate_masks_with_special_tokens_and_transfer_map(ids)
    assert np.array_equal(pr.self_mask[:n, :n], m[0].numpy())
    assert np.array_equal(pr.position_ids[:n], pos[0].numpy())
    assert pr.token_mask[:n].all() and not pr.token_mask[n:].any()
    # padding: identity rows only
    assert np.array_equal(pr.self_mask[n:, n:], np.eye(64 - n, dtype=bool))
    assert not pr.self_mask[:n, n:].any()


@pytest.mark.parametrize("text", PROMPTS)
def test_tokens_match_hf_processor(tok, text):
    from transformers import AutoTokenizer
    ref = AutoTokenizer.from_pretrained("IDEA-Research/grounding-dino-base")
    pr = P.build(text, tok)
    assert list(pr.input_ids[:pr.n_tokens]) == ref(pr.caption).input_ids


def test_pure_python_wordpiece_matches(tok):
    wp = P.Tokenizer(pure_python=True)
    for t in PROMPTS + ["電腦 . 😀 . e-mail . U.S.A."]:
        assert wp.encode(t) == tok.encode(t), t


def test_truncation_drops_whole_phrases(tok):
    text = " . ".join(f"object number {i}" for i in range(40))
    pr = P.build(text, tok)
    assert pr.n_tokens <= 64 and pr.truncated
    assert pr.phrases == [p for p in P.normalize(text)[:-2].split(" . ")][: len(pr.phrases)]
    # every kept phrase's tokens are labelled with its index
    for i in range(len(pr.phrases)):
        assert (pr.phrase_of_token == i).any()


def test_sine_embed_matches_hf():
    from transformers.models.grounding_dino.modeling_grounding_dino import get_sine_pos_embed
    pos = np.array([0, 1, 2, 5, 9], np.int64)
    ref = get_sine_pos_embed(torch.from_numpy(pos)[None, :, None].float(), num_pos_feats=256, exchange_xy=False)[0]
    assert np.allclose(P.sine_embed_1d(pos), ref.numpy(), atol=1e-5)
