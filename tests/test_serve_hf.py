"""HF backend on the tiny random cohere2 fixture (CPU, offline)."""

import pytest

from magpie.config import GenerationConfig
from magpie.serve import Generator, Request, render_chat


def test_chat_render_matches_cohere_layout(tiny_tokenizer):
    text = render_chat(tiny_tokenizer, Request(prompt="hi", assistant_prefix="Sure"))
    assert text.startswith("<BOS_TOKEN><|START_OF_TURN_TOKEN|><|USER_TOKEN|>hi")
    assert text.endswith("<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>Sure")


def test_hf_backend_generates_and_prices(tiny_cohere2, tiny_tokenizer):
    pytest.importorskip("torch")
    from magpie.serve.hf import HFBackend

    backend = HFBackend(tiny_cohere2, tiny_tokenizer, device="cpu", batch_size=2)
    gen = Generator(backend)
    reqs = [Request(prompt="hello there"), Request(prompt="kya haal hai"),
            Request(prompt="x", assistant_prefix="ok")]
    out = gen.generate_priced(reqs, GenerationConfig(max_new_tokens=5))
    assert len(out) == 3
    for (comp, cost), req in zip(out, reqs, strict=True):
        assert 0 < comp.output_tokens <= 5
        assert comp.prompt_tokens > len(req.prompt)  # template tokens are counted
        assert cost.flops == pytest.approx(
            2 * gen.cost_model.P * (comp.prompt_tokens + comp.output_tokens))
    assert out[2][0].full_text.startswith("ok")


def test_greedy_is_deterministic(tiny_cohere2, tiny_tokenizer):
    pytest.importorskip("torch")
    from magpie.serve.hf import HFBackend

    backend = HFBackend(tiny_cohere2, tiny_tokenizer, device="cpu")
    g = GenerationConfig(max_new_tokens=6)
    a = backend.generate([Request(prompt="abc")], g)[0].text
    b = backend.generate([Request(prompt="abc")], g)[0].text
    assert a == b


def test_load_backend_echo_and_errors():
    from magpie.serve import EchoBackend, load_backend

    assert isinstance(load_backend("echo"), EchoBackend)
    with pytest.raises(ValueError):
        load_backend("nope", model="x")
    with pytest.raises(ValueError):
        load_backend("hf")
