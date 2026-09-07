from types import SimpleNamespace

from miniservellm.model_adapter.adapter_factory import create_model_adapter
from miniservellm.model_adapter.gemma4_config import (
    build_gemma4_layer_specs,
    convert_gemma4_config,
)


def make_config():
    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ]
    return SimpleNamespace(
        model_type="gemma4",
        text_config=SimpleNamespace(
            model_type="gemma4_text",
            vocab_size=262144,
            hidden_size=1536,
            intermediate_size=6144,
            num_hidden_layers=6,
            num_attention_heads=8,
            num_key_value_heads=1,
            head_dim=256,
            global_head_dim=512,
            max_position_embeddings=131072,
            rms_norm_eps=1e-6,
            hidden_activation="gelu_pytorch_tanh",
            tie_word_embeddings=True,
            sliding_window=512,
            layer_types=layer_types,
            hidden_size_per_layer_input=256,
            vocab_size_per_layer_input=262144,
            num_kv_shared_layers=2,
            use_double_wide_mlp=True,
            final_logit_softcapping=30.0,
            rope_parameters={
                "sliding_attention": {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                },
                "full_attention": {
                    "rope_type": "proportional",
                    "rope_theta": 1000000.0,
                    "partial_rotary_factor": 0.25,
                },
            },
        ),
    )


def test_gemma4_layer_specs_preserve_heterogeneous_attention():
    specs = build_gemma4_layer_specs(make_config())

    assert len(specs) == 6
    assert specs[0].head_dim == 256
    assert specs[4].head_dim == 512
    assert specs[0].rope_type == "default"
    assert specs[4].rope_type == "proportional"
    assert specs[4].rotary_dim == 128
    assert specs[-1].is_full
    assert specs[-1].use_double_wide_mlp
    # Source mapping is deliberately deferred until the official cache
    # semantics are implemented.
    assert all(spec.kv_source_layer is None for spec in specs)


def test_gemma4_config_conversion_keeps_legacy_and_extended_fields():
    config = convert_gemma4_config(make_config())

    assert config.model_type == "gemma4"
    assert config.hidden_size == 1536
    assert config.num_hidden_layers == 6
    assert config.hidden_size_per_layer_input == 256
    assert config.num_kv_shared_layers == 2
    assert config.final_logit_softcapping == 30.0
    assert len(config.layer_specs) == 6


def test_model_adapter_factory_selects_gemma4():
    adapter = create_model_adapter(make_config())
    assert adapter.__class__.__name__ == "Gemma4Adapter"
