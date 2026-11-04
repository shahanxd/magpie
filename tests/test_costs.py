import math

import pytest

from magpie.costs import TINY_AYA_FIRE, AttackCost, CostModel, ModelShape


def test_tiny_aya_param_count_matches_tech_report():
    # Derived from the published config; the tech report quotes 3.35B.
    assert TINY_AYA_FIRE.params_total == pytest.approx(3.35e9, rel=0.005)
    # Embeddings are tied, so they are counted once: 262144 × 2048.
    assert TINY_AYA_FIRE.params_embedding == 262144 * 2048
    assert TINY_AYA_FIRE.params_non_embedding < TINY_AYA_FIRE.params_total


def test_forward_is_two_p_per_token():
    cm = CostModel(TINY_AYA_FIRE)
    assert cm.forward_flops(1) == pytest.approx(2 * TINY_AYA_FIRE.params_total)
    c = cm.generation(prompt_tokens=400, output_tokens=100, n_samples=3)
    assert c.flops == pytest.approx(2 * TINY_AYA_FIRE.params_total * 500 * 3)
    assert c.queries == 3


def test_training_is_six_p_per_token():
    cm = CostModel(TINY_AYA_FIRE)
    c = cm.finetune(n_examples=10, avg_seq_len=100, epochs=1)
    assert c.train_flops == pytest.approx(6 * TINY_AYA_FIRE.params_total * 1000)
    # LoRA: backward term scaled down, forward kept.
    lora = cm.finetune(n_examples=10, avg_seq_len=100, trainable_fraction=0.0)
    assert lora.train_flops == pytest.approx(2 * TINY_AYA_FIRE.params_total * 1000)


def test_costs_add_and_amortise():
    a = AttackCost(flops=1.0, queries=1, train_flops=10.0)
    b = AttackCost(flops=2.0, queries=2)
    s = a + b
    assert (s.flops, s.queries, s.train_flops) == (3.0, 3, 10.0)
    assert s.total_flops == 13.0
    assert a.amortise(10).train_flops == pytest.approx(1.0)
    assert AttackCost(flops=100.0).log10_flops == pytest.approx(2.0)
    assert AttackCost().log10_flops == 0.0


def test_direction_fit_is_inference_only():
    cm = CostModel(TINY_AYA_FIRE)
    c = cm.direction_fit(n_contrast_prompts=256, avg_seq_len=400)
    assert c.train_flops == 0.0
    assert c.flops == pytest.approx(2 * TINY_AYA_FIRE.params_total * 256 * 400)


def test_non_gated_mlp_shape():
    s = ModelShape("x", n_layers=1, d_model=4, d_mlp=8, n_heads=1, n_kv_heads=1, head_dim=4,
                   vocab_size=10, gated_mlp=False, tied_embeddings=False)
    assert s.mlp_params_per_layer == 2 * 4 * 8
    assert s.params_embedding == 2 * 10 * 4
    assert math.isfinite(CostModel(s).forward_flops(5))
