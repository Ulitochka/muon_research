"""Dataset adapter turning framework interaction CSVs into TIGER seq2seq data.

The framework's sequential-recommendation CSVs (the same files consumed by the
MiniOneRec SFT path) contain, per row:

    history_item_sid : list of concatenated SID strings (chronological history)
    item_sid         : the next item's concatenated SID string (target)

TIGER is an encoder-decoder model, so each row becomes a single seq2seq example:

    encoder input_ids : [ h1_a h1_b h1_c  h2_a h2_b h2_c ...  EOS  PAD ... ]
    decoder labels    : [ t_a t_b t_c EOS ]

The user-ID token used by the original TIGER implementation is intentionally
omitted. Sequences are padded to a fixed length so the plain
``transformers.default_data_collator`` can be used (no HF tokenizer required).
"""

from typing import Dict, List, Optional

import pandas as pd
from datasets import Dataset as HFDataset

from tiger.tokenizer import TigerSidTokenizer


def _rows_to_columns(
    df: pd.DataFrame,
    tokenizer: TigerSidTokenizer,
    with_index: bool,
) -> Dict[str, List]:
    input_ids_all: List[List[int]] = []
    attention_all: List[List[int]] = []
    labels_all: List[List[int]] = []
    idx_all: List[int] = []

    for pos, (_, row) in enumerate(df.iterrows()):
        history_sids = tokenizer.parse_sid_list(row["history_item_sid"])
        if not history_sids:
            # Skip degenerate rows with empty history (nothing to condition on).
            continue
        input_ids, attention_mask = tokenizer.encode_history(history_sids)
        labels = tokenizer.encode_target(str(row["item_sid"]))

        input_ids_all.append(input_ids)
        attention_all.append(attention_mask)
        labels_all.append(labels)
        idx_all.append(pos)

    columns: Dict[str, List] = {
        "input_ids": input_ids_all,
        "attention_mask": attention_all,
        "labels": labels_all,
    }
    if with_index:
        columns["idx"] = idx_all
    return columns


def build_seq2seq_dataset(
    csv_path: str,
    tokenizer: TigerSidTokenizer,
    sample: int = -1,
    seed: int = 42,
    with_index: bool = False,
) -> HFDataset:
    """Build a HuggingFace ``Dataset`` of TIGER seq2seq examples from a CSV.

    Args:
        csv_path: Path to a framework sequential-recommendation CSV.
        tokenizer: A :class:`TigerSidTokenizer`.
        sample: If ``> 0``, randomly subsample this many rows (for quick runs).
        seed: Random seed for subsampling.
        with_index: If ``True``, also emit an ``idx`` column (useful for eval).

    Returns:
        A ``datasets.Dataset`` with ``input_ids``/``attention_mask``/``labels``
        (and optionally ``idx``), formatted as torch tensors.
    """
    df = pd.read_csv(csv_path)
    if sample is not None and sample > 0 and sample < len(df):
        df = df.sample(sample, random_state=seed).reset_index(drop=True)

    columns = _rows_to_columns(df, tokenizer, with_index=with_index)
    dataset = HFDataset.from_dict(columns)
    dataset.set_format(type="torch")
    return dataset


def load_seq2seq_split(
    train_file: Optional[str],
    eval_file: Optional[str],
    tokenizer: TigerSidTokenizer,
    sample: int = -1,
    seed: int = 42,
) -> Dict[str, Optional[HFDataset]]:
    """Convenience helper that builds the train/eval datasets in one call."""
    out: Dict[str, Optional[HFDataset]] = {"train": None, "eval": None}
    if train_file:
        out["train"] = build_seq2seq_dataset(train_file, tokenizer, sample=sample, seed=seed)
    if eval_file:
        out["eval"] = build_seq2seq_dataset(eval_file, tokenizer, sample=sample, seed=seed)
    return out
