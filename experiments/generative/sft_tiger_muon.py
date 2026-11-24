import argparse
import math
import os
import sys
from pathlib import Path
import json
import time

import torch
import torch_npu
import transformers
from torch.optim.lr_scheduler import LambdaLR
from transformers import EarlyStoppingCallback, default_data_collator, set_seed

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_CURRENT_DIR, "../../"))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from experiments.benchmark_tools.time_measurement import TimeDecorator, StepTimerCallback
from experiments.benchmark_tools.plots import parse_loss, plot_metric, save_time_report, save_graphic, parse_metrics, plot_several_metrics
from experiments.benchmark_tools.training_callbacks import _PrinterCallback, EvalCallback

from minionerec_engine.tiger.tokenizer import TigerSidTokenizer  # noqa: E402
from minionerec_engine.tiger.data import build_seq2seq_dataset  # noqa: E402
from minionerec_engine.tiger.model import build_tiger_t5, count_parameters, DEFAULT_T5_CONFIG  # noqa: E402
from Muon.muon import SingleDeviceMuonWithAuxAdam  # noqa: E402

os.environ.setdefault("WANDB_DISABLED", "true")

# adam parameters substrings
_ADAM_NAME_HINTS = ("shared", "lm_head", "embed", "relative_attention_bias", "norm")


def _split_params(model):
    """Route 2D non-embedding weights to Muon, everything else to AdamW."""
    muon_params, adam_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        force_adam = any(hint in name for hint in _ADAM_NAME_HINTS)
        if (not force_adam) and param.ndim == 2:
            muon_params.append(param)
        else:
            adam_params.append(param)
    return muon_params, adam_params


class TimedSingleDeviceMuonWithAuxAdam(SingleDeviceMuonWithAuxAdam):
    """
        SingleDeviceMuonWithAuxAdam with step time measure decorator
    """
    @TimeDecorator
    def step(self, *args, **kwargs):
        return super().step(*args, **kwargs)    


def train(args):
    set_seed(args.seed)

    print(f"=== TIGER SFT (Muon) | category dir: {args.output_dir} ===")

    tokenizer = TigerSidTokenizer(
        index_path=args.index_path,
        codebook_size=args.codebook_size,
        max_item_seq_len=args.max_item_seq_len,
    )
    print(f"Vocab size: {tokenizer.vocab_size} | n_digit: {tokenizer.n_digit} | "
          f"max_token_seq_len: {tokenizer.max_token_seq_len} | pad: {tokenizer.padding_token} | eos: {tokenizer.eos_token}")

    t5_cfg = {
        "num_layers": args.num_layers,
        "num_decoder_layers": args.num_decoder_layers,
        "d_model": args.d_model,
        "d_ff": args.d_ff,
        "num_heads": args.num_heads,
        "d_kv": args.d_kv,
        "dropout_rate": args.dropout_rate,
        "activation_function": args.activation_function,
        "feed_forward_proj": args.feed_forward_proj,
    }
    model = build_tiger_t5(tokenizer, t5_cfg)
    print(count_parameters(model))

    train_data = build_seq2seq_dataset(args.train_file, tokenizer, sample=args.sample, seed=args.seed)
    val_data = build_seq2seq_dataset(args.eval_file, tokenizer, sample=args.sample, seed=args.seed)
    print(f"Train examples: {len(train_data)} | Val examples: {len(val_data)}")

    gradient_accumulation_steps = max(1, args.batch_size // args.micro_batch_size)
    steps_per_epoch = max(1, len(train_data) // args.batch_size)
    total_steps = steps_per_epoch * args.num_epochs

    # ---- Muon + auxiliary AdamW optimizer (framework implementation) ----
    muon_params, adam_params = _split_params(model)
    print(f"Muon params (2D attn/FFN weights): {len(muon_params)} tensors")
    print(f"Adam params (embed/head/norms/1D): {len(adam_params)} tensors")

    param_groups = []
    if muon_params:
        param_groups.append({
            "params": muon_params,
            "use_muon": True,
            "lr": args.muon_lr,
            "momentum": args.muon_momentum,
            "weight_decay": args.muon_wd,
        })
    if adam_params:
        param_groups.append({
            "params": adam_params,
            "use_muon": False,
            "lr": args.adam_lr,
            "betas": (0.9, 0.999),
            "eps": 1e-8,
            "weight_decay": args.adam_wd,
        })
    optimizer = TimedSingleDeviceMuonWithAuxAdam(param_groups)  # SingleDeviceMuonWithAuxAdam just with step time measure decorator

    # Cosine schedule with warmup, applied as a multiplicative factor on each
    # group's base LR (Muon and Adam groups scale together).
    def lr_lambda(current_step):
        if current_step < args.warmup_steps:
            return max(0.0, float(current_step) / float(max(1, args.warmup_steps)))
        progress = float(current_step - args.warmup_steps) / float(max(1, total_steps - args.warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    print("=" * 50)
    print(f"Optimizer: SingleDeviceMuonWithAuxAdam")
    print(f"  Muon  -> lr={args.muon_lr}  momentum={args.muon_momentum}  wd={args.muon_wd}")
    print(f"  AdamW -> lr={args.adam_lr}  wd={args.adam_wd}")
    print("=" * 50)

    training_args = transformers.TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.micro_batch_size,
        per_device_eval_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        num_train_epochs=args.num_epochs,
        warmup_steps=args.warmup_steps,
        bf16=args.bf16,
        logging_steps=args.logging_steps,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False,
    )
    # Debugging: Verify critical training arguments
    print("\n" + "="*40)
    print("DEBUG CHECK: Verifying Training Arguments")
    print(f"--> Received Adam Learning Rate: {args.adam_lr}")
    print(f"--> Received Adam Weight Decay:  {args.adam_wd}")
    print(f"--> Received Muon Learning Rate: {args.muon_lr}")
    print(f"--> Received Muon Weight Decay:  {args.muon_wd}")    
    print("-"*40 + "\n")
    print(f"--> Received Warmup Steps setting:  {training_args.warmup_steps}")
    print(f"--> Received Max Grad Norm setting:  {training_args.max_grad_norm}")
    print(f"--> Received BF16 setting:  {training_args.bf16}")
    print(f"--> Received Number of Epochs setting:  {training_args.num_train_epochs}")
    print(f"--> Received Grad Accumulation Steps setting:  {training_args.gradient_accumulation_steps}")
    print("="*40 + "\n")
      
    # define training callbacks
    topk_list = [1, 3, 5, 10, 20, 50]
    topk_list = [k for k in topk_list if k <= args.num_beams]   # remove top-k greater than num-beams

    eval_callback = EvalCallback(args.eval_file, 
                                 args.output_dir,
                                 tokenizer,
                                 topk_list=topk_list,
                                 patience=args.early_stopping_patience,
                                 seed=args.seed,
                                 num_beams=args.num_beams
                                 )
    step_timer_callback = StepTimerCallback()
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=default_data_collator,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience),
            #_PrinterCallback(),
            step_timer_callback, # measure time on each step
            eval_callback
        ],
        optimizers=(optimizer, lr_scheduler),
    )
    start_time = time.perf_counter()
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    train_exec_time = time.perf_counter() - start_time # full training execution time

    trainer.save_model(args.output_dir) # save best eval-loss checkpoint

    # plot metrics:
    val_losses, val_steps, val_epochs = parse_loss(trainer.state.log_history, loss_type='val')
    train_losses, train_steps, train_epochs = parse_loss(trainer.state.log_history, loss_type='train')

    # loss graphics:
    val_loss_graphic = plot_metric(val_epochs, val_losses, title='Val loss curve')
    train_loss_graphic = plot_metric(train_steps, train_losses, x_label='Steps', title='Train loss curve')
    save_graphic(val_loss_graphic, args.output_dir, 'val_loss.pdf')
    save_graphic(train_loss_graphic, args.output_dir, 'train_loss.pdf')

    # metrics graphics:
    val_metrics = eval_callback.metrics_history
    ndcg_metrics = parse_metrics(val_metrics, 'ndcg', topk_list[:4])    # take only first 4 top-k's
    hr_metrics = parse_metrics(val_metrics, 'hr', topk_list[:4])        #

    ndcg_metrics_graphic = plot_several_metrics(ndcg_metrics)
    hr_metrics_grapic = plot_several_metrics(hr_metrics)

    save_graphic(ndcg_metrics_graphic, args.output_dir, 'ndcg.pdf')
    save_graphic(hr_metrics_grapic, args.output_dir, 'hr.pdf')

    # time report:
    save_time_report(args.output_dir, 
                     optimizer.step.times, 
                     step_timer_callback.times, 
                     eval_callback.times, 
                     train_exec_time
                     )

    # save history and other metadata:
    # val metrics:
    with open(Path(args.output_dir)/'metrics.json', 'w', encoding='utf-8') as f:
        json.dump(val_metrics, f, indent=2)

    # train history:
    with open(Path(args.output_dir)/'train_logs.json', 'w', encoding='utf-8') as f:
        json.dump(trainer.state.log_history, f, indent=2)
    

def _parse_args():
    p = argparse.ArgumentParser(description="TIGER SFT with the Muon optimizer.")
    # data / io
    p.add_argument("--train_file", required=True)
    p.add_argument("--eval_file", required=True)
    p.add_argument("--index_path", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--resume_from_checkpoint", default=None)
    p.add_argument("--sample", type=int, default=-1)
    p.add_argument("--seed", type=int, default=42)
    # training
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--micro_batch_size", type=int, default=256)
    p.add_argument("--num_epochs", type=int, default=200)
    p.add_argument("--num_beams", type=int, default=20)
    p.add_argument("--warmup_steps", type=int, default=20)
    p.add_argument("--early_stopping_patience", type=int, default=20)
    p.add_argument("--logging_steps", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--bf16", type=lambda x: str(x).lower() != "false", default=True)
    # optimizer hyperparameters
    p.add_argument("--muon_lr", type=float, default=1e-2)
    p.add_argument("--muon_wd", type=float, default=0.0)
    p.add_argument("--muon_momentum", type=float, default=0.95)
    p.add_argument("--adam_lr", type=float, default=3e-3)
    p.add_argument("--adam_wd", type=float, default=0.05)
    # semantic-id / sequence
    p.add_argument("--codebook_size", type=int, default=256)
    p.add_argument("--max_item_seq_len", type=int, default=20)
    # T5 architecture
    p.add_argument("--num_layers", type=int, default=DEFAULT_T5_CONFIG["num_layers"])
    p.add_argument("--num_decoder_layers", type=int, default=DEFAULT_T5_CONFIG["num_decoder_layers"])
    p.add_argument("--d_model", type=int, default=DEFAULT_T5_CONFIG["d_model"])
    p.add_argument("--d_ff", type=int, default=DEFAULT_T5_CONFIG["d_ff"])
    p.add_argument("--num_heads", type=int, default=DEFAULT_T5_CONFIG["num_heads"])
    p.add_argument("--d_kv", type=int, default=DEFAULT_T5_CONFIG["d_kv"])
    p.add_argument("--dropout_rate", type=float, default=DEFAULT_T5_CONFIG["dropout_rate"])
    p.add_argument("--activation_function", default=DEFAULT_T5_CONFIG["activation_function"])
    p.add_argument("--feed_forward_proj", default=DEFAULT_T5_CONFIG["feed_forward_proj"])

    return p.parse_args()


if __name__ == "__main__":
    train(_parse_args())
