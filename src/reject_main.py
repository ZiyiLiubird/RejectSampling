import argparse
import json
import os
import copy
import time
from datetime import datetime
import torch
import transformers

from distributedworker import DistributedRollout, rollout

def main(args_dict,):
    current_date = datetime.now().strftime('%Y-%m-%d')
    worker = DistributedRollout(worker_num=args_dict['num_process'])
    samples = json.load(open(args_dict['sample_load_path'], "r"))

    num_process = args_dict['num_process']
    batch = len(samples) // num_process

    args_dict['sample_reward_save_path'] = os.path.join(args_dict['sample_reward_save_path'], current_date)
    if not os.path.exists(args_dict['sample_reward_save_path']):
        os.makedirs(args_dict['sample_reward_save_path'])

    args_dict['sft_save_path'] = os.path.join(args_dict['sft_save_path'], current_date)
    if not os.path.exists(args_dict['sft_save_path']):
        os.makedirs(args_dict['sft_save_path'])

    for i in range(num_process):
        if i == num_process - 1:
            data = samples[i*batch:]
        else:
            data = samples[i*batch: (i+1)*batch]

        args_dict['samples'] = data
        worker.start_rollout(task_id=i, args_dict=copy.deepcopy(args_dict))

    result_dict = {}
    cnt = 0
    while True:
        for i in range(num_process):
            flag = worker.get_results(task_id=i)
            if flag is not None and i not in result_dict:
                result_dict[i] = flag
                cnt += 1
        if cnt == num_process:
            worker.finish()
            break


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--reward_model_path', default="/alg_vepfs/public/models/rm1017", type=str)
    parser.add_argument('--sample_load_path', default='/alg_vepfs/public/LZY/sample_data/2023-11-14/sample_all.json', type=str)
    parser.add_argument('--sample_reward_save_path', default='/alg_vepfs/public/LZY/sample_data', type=str)
    parser.add_argument('--sft_save_path', default='/alg_vepfs/public/LZY/sft_data', type=str)
    parser.add_argument('--batch_size', default=3, type=int)
    parser.add_argument('--model_max_tokens', default=4096, type=int)
    parser.add_argument('--max_context_tokens', default=3500, type=int)
    parser.add_argument('--num_process', default=8, type=int)

    args = parser.parse_args()
    args_dict = vars(args)

    main(args_dict)