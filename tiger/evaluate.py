"""Evaluation for the TIGER adapter: constrained beam search + HR@K / NDCG@K.

Loads a trained TIGER T5 checkpoint, runs (optionally constrained) beam search
over the compact semantic-ID vocabulary, and reports HR@K / NDCG@K using the
same definitions as the framework's ``minionerec_engine/calc.py`` (metrics are
computed at the concatenated semantic-ID level).

Usage (single GPU)::

    python -m minionerec_engine.tiger.evaluate \
        --checkpoint outputs/generative/sft_tiger_adam/Industrial_and_Scientific/final_checkpoint \
        --index_path minionerec_engine/data/Amazon/index/Industrial_and_Scientific.index.json \
        --test_file  minionerec_engine/data/Amazon/test/Industrial_and_Scientific_5_2016-10-2018-11.csv
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch_npu
from tqdm import tqdm
from transformers import T5ForConditionalGeneration

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from tiger.tokenizer import TigerSidTokenizer  # noqa: E402
from tiger.data import build_seq2seq_dataset  # noqa: E402


def set_seed(seed: int):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.npu.is_available():
        torch.npu.manual_seed_all(seed)


@torch.no_grad()
def infer(
    model,
    tokenizer,
    device,
    test_file: str,
    result_json: str = "",
    batch_size: int = 32,
    num_beams: int = 20,
    topk_list:list=[1, 3, 5, 10, 20, 50],
    constrained: bool = True,
    sample: int = -1,
    seed: int = 42,
):
    model.to(device)
    model.eval()
    print(f'eval model device={next(model.parameters()).device}')

    n_digit = tokenizer.n_digit    
    dataset = build_seq2seq_dataset(test_file, tokenizer, sample=sample, seed=seed, with_index=True)

    prefix_fn = tokenizer.build_prefix_allowed_tokens_fn(decoder_start_token_id=0) if constrained else None

    valid_topk = [k for k in topk_list if k <= num_beams]
    hr = np.zeros(len(valid_topk))
    ndcg = np.zeros(len(valid_topk))
    n_examples = 0
    records = []

    for start in tqdm(range(0, len(dataset), batch_size), desc="Evaluating"):
        batch = dataset[start:start + batch_size]
        input_ids = torch.as_tensor(batch["input_ids"]).to(device)
        attention_mask = torch.as_tensor(batch["attention_mask"]).to(device)
        labels = torch.as_tensor(batch["labels"])
        cur_bs = input_ids.shape[0]

        gen = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=n_digit + 1,
            num_beams=num_beams,
            num_return_sequences=num_beams,
            prefix_allowed_tokens_fn=prefix_fn,
            return_dict_in_generate=True,
            output_scores=True,
            use_cache=True,
        )
        seqs = gen.sequences.reshape(cur_bs, num_beams, -1)
        # skip decoder_start token, take the n_digit semantic tokens
        pred_tokens = seqs[:, :, 1:1 + n_digit].cpu().tolist()

        for b in range(cur_bs):
            target = tuple(int(x) for x in labels[b][:n_digit].tolist())
            preds = [tuple(p) for p in pred_tokens[b]]

            rank = None
            for cur_rank, pred in enumerate(preds):
                if pred == target:
                    rank = cur_rank
                    break

            for ki, k in enumerate(valid_topk):
                if rank is not None and rank < k:
                    hr[ki] += 1
                    ndcg[ki] += 1.0 / math.log2(rank + 2)
            n_examples += 1

            if result_json:
                records.append({
                    "target_sid": tokenizer.ids_to_sid_string(target),
                    "predict": [tokenizer.ids_to_sid_string(p) for p in preds],
                    "hit_rank": rank,
                })

    hr = hr / max(1, n_examples)
    ndcg = ndcg / max(1, n_examples)

    return hr, ndcg, n_examples, records


@torch.no_grad()
def evaluate(   # model+tokenizer eval on test_file
    model,
    tokenizer,
    test_file: str,
    result_json: str = "",
    batch_size: int = 32,
    num_beams: int = 20,
    topk_list = [1, 3, 5, 10, 20, 50],
    constrained: bool = True,
    sample: int = -1,
    seed: int = 42,
    device_id: int = 0,
):
    set_seed(seed)
    device = torch.device(f"npu:{device_id}" if torch.npu.is_available() else "cpu")
    topk_list = [k for k in topk_list if k <= num_beams]

    return infer(
                model,
                tokenizer,
                device,
                test_file,
                result_json,
                batch_size,
                num_beams,
                topk_list,
                constrained,
                sample,
                seed,
            )


@torch.no_grad()
def evaluate_checkpoint(
    checkpoint: str,
    index_path: str,
    test_file: str,
    result_json: str = "",
    batch_size: int = 32,
    num_beams: int = 20,
    max_item_seq_len: int = 20,
    codebook_size: int = 256,
    constrained: bool = True,
    sample: int = -1,
    seed: int = 42,
    device_id: int = 0,
):
    set_seed(seed)
    device = torch.device(f"npu:{device_id}" if torch.npu.is_available() else "cpu")

    tokenizer = TigerSidTokenizer(
        index_path=index_path,
        codebook_size=codebook_size,
        max_item_seq_len=max_item_seq_len,
    )
    model = T5ForConditionalGeneration.from_pretrained(checkpoint)
    topk_list = [1, 3, 5, 10, 20, 50]
    valid_topk = [k for k in topk_list if k <= num_beams]

    hr, ndcg, n_examples, records = infer(model,
                                          tokenizer,
                                          device,
                                          test_file,
                                          result_json,
                                          batch_size,
                                          num_beams,
                                          topk_list,
                                          constrained,
                                          sample,
                                          seed
                                         )
    print("\n" + "=" * 50)
    print(f" TIGER Evaluation  (constrained={constrained}, beams={num_beams})")
    print("=" * 50)
    print(f"Examples evaluated: {n_examples}")
    print(f"{'K':>4} | {'HR@K':>10} | {'NDCG@K':>10}")
    for ki, k in enumerate(valid_topk):
        print(f"{k:>4} | {hr[ki]:>10.4f} | {ndcg[ki]:>10.4f}")
    print("=" * 50 + "\n")

    if result_json:
        os.makedirs(os.path.dirname(os.path.abspath(result_json)), exist_ok=True)
        with open(result_json, "w", encoding="utf-8") as f:
            json.dump({
                "config": {
                    "checkpoint": checkpoint,
                    "num_beams": num_beams,
                    "constrained": constrained,
                    "n_examples": n_examples,
                },
                "metrics": {
                    f"HR@{k}": float(hr[ki]) for ki, k in enumerate(valid_topk)
                } | {
                    f"NDCG@{k}": float(ndcg[ki]) for ki, k in enumerate(valid_topk)
                },
                "records": records,
            }, f, indent=2)
        print(f"Results saved to: {result_json}")

    return {"HR": hr.tolist(), "NDCG": ndcg.tolist(), "topk": valid_topk}


def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate a TIGER checkpoint (HR@K / NDCG@K).")
    p.add_argument("--checkpoint", required=True, help="Path to a trained TIGER T5 checkpoint dir.")
    p.add_argument("--index_path", required=True, help="Framework *.index.json path.")
    p.add_argument("--test_file", required=True, help="Framework test CSV path.")
    p.add_argument("--result_json", default="", help="Optional path to dump per-example predictions.")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_beams", type=int, default=20)
    p.add_argument("--max_item_seq_len", type=int, default=20)
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--constrained", type=lambda x: str(x).lower() != "false", default=True)
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device_id", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    evaluate_checkpoint(
        checkpoint=args.checkpoint,
        index_path=args.index_path,
        test_file=args.test_file,
        result_json=args.result_json,
        batch_size=args.batch_size,
        num_beams=args.num_beams,
        max_item_seq_len=args.max_item_seq_len,
        codebook_size=args.codebook_size,
        constrained=args.constrained,
        sample=args.sample,
        seed=args.seed,
        device_id=args.device_id,
    )
