"""Hardening code on the tiny random cohere2 model (CPU, offline, a few steps)."""

import pytest

from magpie.defenses.harden import RRConfig, _collate, scaled_layers, train_circuit_breakers


def test_scaled_layers_maps_paper_layers_to_tiny_aya_depth():
    assert scaled_layers(36) == [11, 22]
    assert scaled_layers(32) == [10, 20]
    assert all(1 <= layer < 2 for layer in scaled_layers(2))


def _rows(tok, texts, response_from=None, width=24):
    rows = []
    for t in texts:
        ids = tok(t, add_special_tokens=False)["input_ids"][:width]
        ids = ids + [tok.pad_token_id] * (width - len(ids))
        att = [1 if x != tok.pad_token_id else 0 for x in ids]
        labels = list(ids) if response_from is None else (
            [-100] * response_from + ids[response_from:])
        labels = [lb if a else -100 for lb, a in zip(labels, att, strict=True)]
        rows.append({"input_ids": ids, "attention_mask": att, "labels": labels})
    return rows


def test_collate_right_pads_and_masks(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    rows = [{"input_ids": [5, 6, 7]}, {"input_ids": [5]}]
    b = _collate(rows, pad_id=0, device=torch.device("cpu"))
    assert b["input_ids"].tolist() == [[5, 6, 7], [5, 0, 0]]
    assert b["attention_mask"].tolist() == [[1, 1, 1], [1, 0, 0]]
    assert b["labels"][1].tolist() == [5, -100, -100]


def test_rr_training_runs_changes_weights_and_saves(tiny_cohere2, tiny_tokenizer, tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    pytest.importorskip("safetune")
    import copy

    model = copy.deepcopy(tiny_cohere2)
    before = {k: v.clone() for k, v in model.state_dict().items()}
    retain = _rows(tiny_tokenizer, ["sample text gamma", "sample text delta"])
    harmful = _rows(tiny_tokenizer, ["sample text alpha one", "sample text beta two"],
                    response_from=6)
    cfg = RRConfig(steps=3, batch_size=2, fp16=False, grad_checkpointing=False,
                   target_layers=[1], log_every=0)
    res = train_circuit_breakers(model, tiny_tokenizer, retain, harmful, tmp_path / "rr", cfg)
    assert len(res.losses) == 3 and all(x == x for x in res.losses)  # finite, not NaN
    assert res.target_layers == [1]
    # Saved, reloadable, and the merged weights differ from the original.
    from transformers import AutoModelForCausalLM

    back = AutoModelForCausalLM.from_pretrained(tmp_path / "rr")
    changed = any(not torch.equal(before[k], v) for k, v in back.state_dict().items()
                  if k in before and v.shape == before[k].shape)
    assert changed
    assert (tmp_path / "rr" / "rr_training.json").is_file()


def test_repnoise_wrapper_trains_and_saves(tiny_cohere2, tiny_tokenizer, tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("safetune")
    import copy

    import safetune
    from datasets import Dataset

    from magpie.defenses.harden import train_repnoise

    safetune.configure(device="cpu", dtype="float32", seed=0)
    tok = copy.deepcopy(tiny_tokenizer)
    tok.name_or_path = "tiny"
    ds_a = Dataset.from_list(_rows(tok, ["sample text one", "sample text two",
                                         "sample text three", "sample text four"]))
    ds_b = Dataset.from_list(_rows(tok, ["sample text five", "sample text six",
                                         "sample text seven", "sample text eight"]))
    out = train_repnoise(copy.deepcopy(tiny_cohere2), tok, tmp_path / "rn", train_dataset=ds_a,
                         harmful_dataset=ds_b, safety_dataset=ds_a, epochs=1, batch_size=2,
                         fp16=False)
    assert (tmp_path / "rn").is_dir() and str(out)
