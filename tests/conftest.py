"""Shared CPU fixtures.

The model fixture is a *tiny random cohere2* network (Tiny Aya's architecture,
two layers, hidden 32), paired with an offline character-level tokenizer that
carries a Cohere-style chat template. No network, no HF token, no GPU -- the
same approach SafeTune's own test-suite uses for Tiny Aya.
"""

from __future__ import annotations

import pytest

SPECIALS = [
    "<PAD>", "<UNK>", "<BOS_TOKEN>", "<EOS_TOKEN>",
    "<|START_OF_TURN_TOKEN|>", "<|END_OF_TURN_TOKEN|>",
    "<|USER_TOKEN|>", "<|CHATBOT_TOKEN|>", "<|SYSTEM_TOKEN|>",
]
CHARS = list(" abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,!?'\"-:;()\n")

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for m in messages %}"
    "<|START_OF_TURN_TOKEN|>"
    "{% if m['role'] == 'user' %}<|USER_TOKEN|>"
    "{% elif m['role'] == 'system' %}<|SYSTEM_TOKEN|>"
    "{% else %}<|CHATBOT_TOKEN|>{% endif %}"
    "{{ m['content'] }}<|END_OF_TURN_TOKEN|>"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{% endif %}"
)


@pytest.fixture(scope="session")
def tiny_tokenizer():
    pytest.importorskip("transformers")
    from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {t: i for i, t in enumerate(SPECIALS + CHARS)}
    core = Tokenizer(models.WordLevel(vocab=vocab, unk_token="<UNK>"))
    core.pre_tokenizer = pre_tokenizers.Split(Regex(r"[\s\S]"), behavior="isolated")
    core.decoder = decoders.Fuse()
    tok = PreTrainedTokenizerFast(
        tokenizer_object=core,
        bos_token="<BOS_TOKEN>",
        eos_token="<|END_OF_TURN_TOKEN|>",
        pad_token="<PAD>",
        unk_token="<UNK>",
        additional_special_tokens=SPECIALS[4:],
    )
    tok.chat_template = CHAT_TEMPLATE
    return tok


@pytest.fixture(scope="session")
def tiny_cohere2(tiny_tokenizer):
    torch = pytest.importorskip("torch")
    from transformers import Cohere2Config, Cohere2ForCausalLM

    torch.manual_seed(0)
    cfg = Cohere2Config(
        vocab_size=len(tiny_tokenizer),
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        sliding_window=64,
        layer_types=["sliding_attention", "full_attention"],
        max_position_embeddings=512,
        initializer_range=0.2,
        pad_token_id=tiny_tokenizer.pad_token_id,
        bos_token_id=tiny_tokenizer.bos_token_id,
        eos_token_id=tiny_tokenizer.eos_token_id,
    )
    model = Cohere2ForCausalLM(cfg).eval()
    return model
