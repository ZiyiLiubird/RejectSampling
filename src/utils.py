from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset
import transformers
import json
import openai

openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"

def get_chat(prompt: str, max_tokens=None, temperature: float = 0,
             stop_strs = None, do_sample=True) -> str:
    response = openai.Completion.create(
        model="/alg_vepfs/public/models/online_model/joyland/llama-13b-4k-teatime-mix-bluemoon_pretrain20231012-0226",
        prompt=prompt,
        stop=stop_strs,
        temperature=temperature,
        do_sample=do_sample,
        max_tokens=max_tokens,
    )
    return response.choices[0]["text"]


