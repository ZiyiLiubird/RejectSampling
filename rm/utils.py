import os
from pathlib import Path

from datasets import Dataset, interleave_datasets, load_dataset
from transformers import AutoTokenizer

from strategy.deepspeed import DeepspeedStrategy

DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_UNK_TOKEN = "<unk>"


def get_strategy(args):
    # default args for deepspeed
    if "seed" not in args:
        args.seed = 42
    if "max_norm" not in args:
        args.max_norm = 1.0
    if "micro_train_batch_size" not in args:
        args.micro_train_batch_size = 1
    if "train_batch_size" not in args:
        args.train_batch_size = 8
    if "local_rank" not in args:
        args.local_rank = -1
    if "bf16" not in args:
        args.bf16 = True
    if "inference_tp_size" not in args:
        args.inference_tp_size = 1
    if "adam_offload" not in args:
        args.adam_offload = False
    if "zpg" not in args:
        args.zpg = 8
    # max_out_tokens for DS inference
    if "max_len" in args and args.max_len is not None:
        args.max_out_tokens = args.max_len
    elif "generate_max_len" in args and "prompt_max_len" in args:
        args.max_out_tokens = args.prompt_max_len + args.generate_max_len
    else:
        args.max_out_tokens = 2048

    strategy = DeepspeedStrategy(
        seed=args.seed,
        max_norm=args.max_norm,
        micro_train_batch_size=args.micro_train_batch_size,
        train_batch_size=args.train_batch_size,
        zero_stage=args.zero_stage,
        max_out_tokens=args.max_out_tokens,
        inference_tp_size=args.inference_tp_size,
        args=args,
    )

    return strategy

