from magpie.config import MODELS, GenerationConfig, RunConfig, git_sha, provenance
from magpie.costs import AttackCost
from magpie.types import Attempt, Language, PromptRecord


def test_attempt_roundtrip_and_uid_stability():
    a = Attempt(prompt_id="p1", model="fire", attack="l0_direct", level="L0", form="en",
                prompt="hi", response="ok", params={"k": 1}, cost=AttackCost(flops=5.0),
                cum_cost=AttackCost(flops=5.0, queries=1))
    d = a.to_dict()
    assert d["cost"]["total_flops"] == 5.0
    b = Attempt.from_dict(d)
    assert b.uid == a.uid
    assert b.cum_cost.queries == 1
    c = Attempt.from_dict({**d, "attempt_index": 1})
    assert c.uid != a.uid


def test_prompt_record_ignores_unknown_keys():
    r = PromptRecord.from_dict({"prompt_id": "x", "text": "t", "form": "hing", "junk": 1})
    assert r.form == Language.HING
    assert r.is_harmful is True


def test_run_config_yaml_and_hash(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("name: t\nmodels: [fire, global]\ngeneration:\n  max_new_tokens: 64\n")
    cfg = RunConfig.from_yaml(p)
    assert cfg.models == ["fire", "global"]
    assert isinstance(cfg.generation, GenerationConfig)
    assert cfg.generation.max_new_tokens == 64 and cfg.generation.greedy
    h1 = cfg.config_hash
    cfg.seed = 1
    assert cfg.config_hash != h1


def test_model_registry_has_kaggle_mirror():
    for key in ("fire", "global", "base"):
        assert MODELS[key]["kaggle_source"].startswith("cohere-labs/tiny-aya/transformers/")
        assert MODELS[key]["hf_id"].startswith("CohereLabs/tiny-aya-")


def test_provenance_env_override(monkeypatch):
    monkeypatch.setenv("MAGPIE_GIT_SHA", "abc1234")
    monkeypatch.setenv("MAGPIE_GIT_DIRTY", "1")
    assert git_sha() == ("abc1234", True)
    prov = provenance(RunConfig(name="x"))
    assert prov["git_sha"] == "abc1234" and prov["config_name"] == "x"
    assert "python" in prov
