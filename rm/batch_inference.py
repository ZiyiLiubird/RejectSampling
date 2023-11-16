import argparse
import os
from datetime import timedelta

import torch
from torch import distributed as dist
from tqdm import tqdm

from models.model import LlamaModelForScore
