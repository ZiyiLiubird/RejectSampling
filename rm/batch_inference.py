import argparse
import os
from datetime import timedelta

import torch
from torch import distributed as dist
from tqdm import tqdm
from utils import get_strategy
from models.rm_model import LlamaModelForScore


def batch_inference():
    # configure strategy
    strategy = get_strategy(args)
    strategy.setup_distributed(timeout=timedelta(seconds=9999999))





def batch_rm_inference():
    pass




def decesion_transformer_processor(args, objs):
    pass



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval_task", type=str, default=None)
    parser.add_argument("--zero_stage", type=int, default=0)
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank for deepspeed")
    parser.add_argument("--greedy_sampling", action="store_true", default=False)
    parser.add_argument("--temperature", type=float, default=1.2)
    parser.add_argument("--flash_attn", action="store_true", default=False)

    # batch inference
    parser.add_argument("--micro_batch_size", type=int, default=16)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--dataset_probs", type=str, default="1.0")
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--max_samples", type=int, default=500000)

    # Decision Transformer training
    parser.add_argument("--post_processor", type=str, default=None)

    # Decision Transformer inference
    parser.add_argument("--enable_dt", action="store_true", default=False)
    parser.add_argument("--dt_prompt", type=str, default="<rm_score>: 5.00", help="decision transformer prompt")

    args = parser.parse_args()
    if args.eval_task and args.eval_task == "generate":
        batch_inference(args)
    elif args.eval_task and args.eval_task == "rm":
        batch_rm_inference(args)
    else:
        print("Invalid or missing '--eval_task' argument. Please specify either 'generate' or 'rm'.")
