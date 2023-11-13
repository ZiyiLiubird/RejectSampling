import argparse
import json
import os
import copy
import time
from datetime import datetime
import torch
import transformers
from vllm import LLM, SamplingParams

from models.model import LlamaModelForScore, get_single_reward_from_model
from rollout import Rollout
from reject import Reject


def main(args_dict,):
    raw_data = json.load(open(args_dict['data_path'], "r"))
    device = torch.device("cuda:0")
    model_tokenizer = transformers.AutoTokenizer.from_pretrained(args_dict['model_path'],
                                                           trust_remote_code=True,
                                                           use_fast=False, device_map=device)
    model = LLM(model=args_dict['model_path'], tokenizer=args_dict['model_path'], trust_remote_code=True,
                max_num_batched_tokens=args_dict['model_max_tokens'],
                tensor_parallel_size=1)

    current_date = datetime.now().strftime('%Y-%m-%d')

    sample_save_path = os.path.join(args_dict['sample_save_path'], current_date)
    if not os.path.exists(sample_save_path):
        os.makedirs(sample_save_path)
    args_dict['raw_data'] = raw_data[:args_dict['sample_num']]
    args_dict['sample_save_path'] = os.path.join(sample_save_path, "sample3.json")

    sample_reward_save_path = os.path.join(args_dict['sample_reward_save_path'], current_date)
    if not os.path.exists(sample_reward_save_path):
        os.makedirs(sample_reward_save_path)
    args_dict['sample_reward_save_path'] = os.path.join(sample_reward_save_path, "sample_reward2.json")

    rollout = Rollout(raw_data=args_dict['raw_data'], model_tokenizer=model_tokenizer, max_context_tokens=args_dict['max_context_tokens'],
                      temperature=args_dict['temperature'],
                      do_sample=args_dict['do_sample'], sample_k=args_dict['sample_k'],
                      save_path=args_dict['sample_save_path'], model=model)

    print(f"Start preprocess")
    rollout.preprocess()
    print(f"Start rollout")
    rollout.rollout()
    print(f"Start save samples")
    samples = rollout.save_samples()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='/alg_vepfs/public/models/online_model/joyland/llama-13b-4k-teatime-mix-bluemoon_pretrain20231012-0226', type=str)
    parser.add_argument('--reward_model_path', default="/alg_vepfs/public/models/rm1017", type=str)
    parser.add_argument('--data_path', default='/alg_vepfs/public/datasets/joyland/7days/7days15k_2.json', type=str)
    parser.add_argument('--sample_save_path', default='/alg_vepfs/public/LZY/sample_data', type=str)
    parser.add_argument('--sample_load_path', default='/alg_vepfs/public/LZY/sample_data/2023-11-09/sample1.json', type=str)
    parser.add_argument('--sample_reward_save_path', default='/alg_vepfs/public/LZY/sample_data', type=str)
    parser.add_argument('--sft_save_path', default='/alg_vepfs/public/LZY/sft_data', type=str)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--model_max_tokens', default=4096, type=int)
    parser.add_argument('--max_context_tokens', default=3900, type=int)
    parser.add_argument('--temperature', default=1.2, type=float)
    parser.add_argument('--do_sample', action='store_true', default=True)
    parser.add_argument('--sample_k', default=5, type=int)
    parser.add_argument('--sample_num', default=15000, type=int)


    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)