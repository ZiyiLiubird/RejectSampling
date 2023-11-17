from multiprocessing import Process, Manager, Queue
from time import sleep
import time
import json

from datetime import datetime
import os
import torch
import transformers
from reject import Reject
from models.model import LlamaModelForScore



def rollout(task_id, args_dict):
    print(f"Process {task_id} start working...")

    sample_reward_save_path = args_dict['sample_reward_save_path']
    args_dict['sample_reward_save_path'] = os.path.join(sample_reward_save_path, f"sample_reward{task_id}.json")

    sft_save_path = args_dict['sft_save_path']
    args_dict['sft_save_path'] = os.path.join(sft_save_path, f"sft{task_id}.json")
    args_dict['postprocess_sft_save_path'] = os.path.join(args_dict['postprocess_sft_save_path'], f"postprocess_sft{task_id}.json")

    reward_device = torch.device(f"cuda:{task_id}")
    reward_model = LlamaModelForScore.from_pretrained(args_dict['reward_model_path']).to(reward_device)
    reward_tokenizer = transformers.AutoTokenizer.from_pretrained(args_dict['reward_model_path'], device_map=reward_device)

    reject = Reject(reward_model=reward_model, reward_tokenizer=reward_tokenizer, max_context_tokens=args_dict['max_context_tokens'],
                    sample_data=args_dict['samples'], batch_size=args_dict['batch_size'],
                    result_save_path=args_dict['sft_save_path'], postprocess_result_path=args_dict['postprocess_sft_save_path'],
                    sample_save_path=args_dict['sample_reward_save_path'],
                    model_device=reward_device)
    reject.batch_reject()
    reject.save()
    reject.sft_postprocess()

    return True


class DistributedRollout:
    def __init__(self, worker_num,):
        self.worker_num = worker_num
        self.result_dict = Manager().dict()
        self.task_queue = Queue()
        self.processors = []
        print(f"Using {worker_num} workers")
        for _ in range(self.worker_num): 
            p = Process(target=self.worker, args=(self.task_queue, self.result_dict))
            p.start()
            self.processors.append(p)

    def worker(self, task_queue, result_dict):
        while True:
            task = task_queue.get()
            if task is None:
                break
            task_id, args_dict = task
            result = rollout(task_id=task_id, args_dict=args_dict)
            result_dict[task_id] = result

    def start_rollout(self, task_id, args_dict,):
        self.task_queue.put((task_id, args_dict))

    def finish(self):
        for _ in range(self.worker_num):  # 向队列中发送结束信号
            self.task_queue.put(None)
        for p in self.processors:  # 等待所有工作进程结束
            p.join()

    def get_results(self, task_id):
        result = self.result_dict.get(task_id, None)
        if result is not None:
            del self.result_dict[task_id]
        return result

