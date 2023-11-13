import openai
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


 
# completion = openai.Completion.create(model="/alg_vepfs/public/models/online_model/joyland/llama-13b-4k-teatime-mix-bluemoon_pretrain20231012-0226",
#                                       prompt="Who is Jack Ma,",
#                                       temperature=0,
#                                     #   max_tokens=4096,
#                                       do_sample=False,
#                                       stop=None)
# completion = get_chat("Who is Jack Ma?")
# print("Completion result:", completion)

# current_date = datetime.now().strftime('%Y-%m-%d')
# print(current_date)

# save_path = "test.json"
# output_data_list = [{"A":0}, {"B":1}]
# for i in range(19):
#     tlist = []
#     for t in range(3):
#         tlist.append({"A":2})
#     output_data_list.append(tlist)

# print(output_data_list)


# with open(save_path, mode='w') as f:
#     json.dump(output_data_list, f, indent=4)
# device = torch.device('cuda:1')
# model_path = '/alg_vepfs/public/LZY/mycodes/models/agentlm-13b'
# model = LLM(model=model_path, tokenizer=model_path, trust_remote_code=True, max_num_batched_tokens=16, device=device)

import numpy as np

string_list = [["apple", "banana", "cherry", "orange"], ["apple", "banana", "cherry", "orange"], ["apple", "banana", "cherry", "orange"]]

# 将字符串列表转换为numpy数组
string_array = np.array(string_list)

# 使用numpy数组的切片操作
sliced_array = string_array[:2]
print(str(sliced_array[0][1]))
print(sliced_array)  # 输出 ['orange']
# input_ids = torch.zeros((10, 1000))
# pad_len = 4096 - input_ids.size(1)
# input_ids = torch.cat([input_ids, torch.zeros((10, pad_len), dtype=torch.long)], dim=1)
# print(input_ids.shape)
# attention_mask = (input_ids != 0).bool()
# print(attention_mask.shape)
