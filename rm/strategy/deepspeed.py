import os
import random
import shutil
from abc import ABC
from collections import defaultdict
from datetime import timedelta
from typing import List, Tuple, Union

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from peft import PeftModel
from torch import distributed as dist
from torch.optim import Optimizer
from torch.utils.data import DataLoader, DistributedSampler


class DeepspeedStrategy(ABC):
    """
    The strategy for training with Accelerator.
    """
    def __init__(self,
                 seed: int = 42,
                 max_norm: float = 0.0,
                 micro_train_batch_size = 1,
                 train_batch_size = 1,
                 zero_stage=2,
                 max_out_tokens=512,
                 inference_tp_size=1,
                 bf16=True,
                 args=None,
                 ) -> None:
        super().__init__()
        self.args = args
        self.stage = zero_stage
        self.train_batch_size = train_batch_size
        self.max_out_tokens = max_out_tokens
        self.micro_train_batch_size = micro_train_batch_size
        self.inference_tp_size = inference_tp_size
        self.bf16 = bf16
        self.adam_offload = args.adam_offload
        self.is_rlhf = False
        self.zpg = args.zpg
        self.seed = seed
        self.max_norm = max_norm
        self.time_steps = defaultdict(int)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def setup_distributed(self, timeout=timedelta(minutes=30))-> None:
        self.set_seed(self.seed)
        if self.args.local_rank == -1 and "LOCAL_RANK" in os.environ:  # for slurm
            self.args.local_rank = int(os.environ["LOCAL_RANK"])

        if self.args.local_rank != -1:
            torch.cuda.set_device(self.args.local_rank)

        deepspeed.init_distributed(timeout=timeout)
        self.world_size = dist.get_world_size()
        self.accumulated_gradient = self.train_batch_size // self.micro_train_batch_size // self.world_size

    def setup_dataloader(self,
                         replay_buffer,
                         batch_size:int,
                         pin_memory:bool=False,
                         shuffle=True,
                         collate_fn=None,
                         drop_last=True,
                         ):
        # DDP only mode, replay buffers on each rank are different.
        sampler = DistributedSampler(
            replay_buffer,
            num_replicas=dist.get_world_size(),
            rank=dist.get_rank(),
            shuffle=shuffle,
            seed=self.seed,
            drop_last=drop_last,
        )
        return DataLoader(replay_buffer, batch_size=batch_size, sampler=sampler, drop_last=drop_last,
                          collate_fn=collate_fn, pin_memory=pin_memory)

    def ds_init_train_model(self, model, optim, scheduler):
        is_actor = isinstance()