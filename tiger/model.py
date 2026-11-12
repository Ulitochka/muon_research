"""Faithful TIGER (T5 encoder-decoder) model builder for MuonRec-E447.

Mirrors the architecture from ``MemGen-GR/genrec/models/TIGER/model.py``: a small
``T5ForConditionalGeneration`` whose vocabulary is the compact semantic-ID vocab
produced by :class:`minionerec_engine.tiger.tokenizer.TigerSidTokenizer`.

The model is a standard HuggingFace ``T5ForConditionalGeneration``; when called
with ``labels`` it returns ``outputs.loss`` (cross-entropy), which lets us drive
training with ``transformers.Trainer`` and either optimizer (AdamW or Muon).
"""

from typing import Dict

from transformers import T5Config, T5ForConditionalGeneration

from tiger.tokenizer import TigerSidTokenizer


# Defaults taken from MemGen-GR/genrec/models/TIGER/config.yaml.
DEFAULT_T5_CONFIG: Dict = {
    "num_layers": 4,
    "num_decoder_layers": 4,
    "d_model": 128,
    "d_ff": 1024,
    "num_heads": 6,
    "d_kv": 64,
    "dropout_rate": 0.1,
    "activation_function": "relu",
    "feed_forward_proj": "relu",
}


def build_tiger_t5(tokenizer: TigerSidTokenizer, cfg: Dict = None) -> T5ForConditionalGeneration:
    """Instantiate a randomly-initialized TIGER-style T5 model.

    Args:
        tokenizer: Provides ``vocab_size``, ``padding_token``, ``eos_token`` and
            ``max_token_seq_len``.
        cfg: Optional overrides for the T5 architecture (see
            :data:`DEFAULT_T5_CONFIG`).

    Returns:
        An untrained ``T5ForConditionalGeneration``.
    """
    merged = dict(DEFAULT_T5_CONFIG)
    if cfg:
        merged.update({k: v for k, v in cfg.items() if v is not None})

    t5config = T5Config(
        num_layers=merged["num_layers"],
        num_decoder_layers=merged["num_decoder_layers"],
        d_model=merged["d_model"],
        d_ff=merged["d_ff"],
        num_heads=merged["num_heads"],
        d_kv=merged["d_kv"],
        dropout_rate=merged["dropout_rate"],
        activation_function=merged["activation_function"],
        vocab_size=tokenizer.vocab_size,
        pad_token_id=tokenizer.padding_token,
        eos_token_id=tokenizer.eos_token,
        decoder_start_token_id=0,
        feed_forward_proj=merged["feed_forward_proj"],
        n_positions=tokenizer.max_token_seq_len,
    )
    model = T5ForConditionalGeneration(config=t5config)
    return model


def count_parameters(model) -> str:
    total = sum(p.numel() for p in model.parameters() if p.requires_grad)
    emb = sum(p.numel() for p in model.get_input_embeddings().parameters() if p.requires_grad)
    return (
        f"#Embedding parameters: {emb}\n"
        f"#Non-embedding parameters: {total - emb}\n"
        f"#Total trainable parameters: {total}\n"
    )
