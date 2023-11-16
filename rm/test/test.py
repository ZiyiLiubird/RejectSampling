import os
import sys
from collections import defaultdict
import copy
import openai
import json
from tenacity import (
    retry,
    stop_after_attempt, # type: ignore
    wait_random_exponential, # type: ignore
)
from typing import Optional, List


openai.api_base = "http://localhost:8000/v1"
openai.api_key = "EMPTY"

# MODEL = "/alg_vepfs/public/LZY/mycodes/safe-rlhf/outputs/1"
MODEL = "/alg_vepfs/public/LZY/mycodes/models/Llama-2-7b-chat-hf"
# MODEL = "/alg_vepfs/public/LZY/mycodes/models/agentlm-13b"

@retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
def get_completion(prompt: str, temperature: float = 0.0, max_tokens: int = 256, stop_strs: Optional[List[str]] = None) -> str:
    response = openai.Completion.create(
        model=MODEL,
        prompt=prompt,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=1,
        frequency_penalty=0.0,
        presence_penalty=0.0,
        stop=stop_strs,
    )
    return response.choices[0].text


print(get_completion(prompt="Hello, who are you?"))