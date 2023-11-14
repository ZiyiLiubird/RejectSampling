import openai
import os
# Modify OpenAI's API key and API base to use vLLM's API server.
openai.api_key = "EMPTY"
openai.api_base = "http://localhost:8000/v1"
from datetime import datetime
import json
import torch
from vllm import LLM, SamplingParams


def get_chat(prompt: str, max_tokens=None, temperature: float = 0,
             stop_strs = None) -> str:
    response = openai.Completion.create(
        model="/alg_vepfs/public/models/online_model/joyland/llama-13b-4k-teatime-mix-bluemoon_pretrain20231012-0226",
        prompt=prompt,
        stop=stop_strs, 
        temperature=temperature,
        do_sample=False,
        max_tokens=max_tokens,
    )
    return response.choices[0]["text"]


save_path = os.path.join('/alg_vepfs/public/datasets/joyland/7days/', '3736sample', '7days3k_0.json')

raw_data1 = json.load(open(save_path, "r"))

print(len(raw_data1))