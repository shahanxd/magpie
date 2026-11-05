"""L3 abliteration on the tiny cohere2 model (CPU; needs safetune)."""

import pytest

from magpie.attacks import ATTACKS, L3Abliteration
from magpie.attacks.base import AttackRunner
from magpie.config import GenerationConfig
from magpie.judges import LexiconJudge
from magpie.types import Level, PromptRecord


def test_registered():
    assert ATTACKS["l3_abliteration"] is L3Abliteration
    assert L3Abliteration().level is Level.L3


def test_fit_requires_hf_backend():
    from magpie.serve import EchoBackend, Generator

    atk = L3Abliteration(pick_layer=1, n_contrast=2)
    with pytest.raises(RuntimeError, match="HF backend"):
        atk.fit(Generator(EchoBackend()), [])


def test_abliteration_ablates_and_reverts(tiny_cohere2, tiny_tokenizer):
    pytest.importorskip("torch")
    pytest.importorskip("safetune")
    import copy

    import safetune

    from magpie.costs import TINY_AYA_FIRE, CostModel
    from magpie.serve.base import Generator
    from magpie.serve.hf import HFBackend

    safetune.configure(device="cpu", dtype="float32", calib_n=4)
    be = HFBackend(copy.deepcopy(tiny_cohere2), tiny_tokenizer, device="cpu", batch_size=4)
    gen = Generator(be, CostModel(TINY_AYA_FIRE))
    atk = L3Abliteration(pick_layer=1, n_contrast=4)
    atk._contrast = lambda tok: (["harmful a", "harmful b", "harmful c", "harmful d"],
                                 ["benign a", "benign b", "benign c", "benign d"])
    recs = [PromptRecord(prompt_id=f"h{i}", text=f"do thing {i}", is_harmful=True)
            for i in range(3)]
    run = AttackRunner(gen, LexiconJudge(), GenerationConfig(max_new_tokens=5)).run(atk, recs)
    assert run.fit_cost.flops > 0           # the direction fit cost is logged
    assert len(run.outcomes) == 3
    assert all(o.n_attempts == 1 for o in run.outcomes)  # one gen per prompt post-ablation
    assert atk._attack is None              # revert() ran
