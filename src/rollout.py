import json
from typing import Dict
import openai

import transformers
import torch
import numpy as np
from tqdm import tqdm, trange
from models.utils import ChatRequire
from models.utils import Message
from models.prepost import Vicuna11SepStyle, AlpacaSepStyle, Conversations 
from models.model import LlamaModelForScore, get_single_reward_from_model
from vllm import SamplingParams


class Rollout:
    def __init__(self, raw_data, model_tokenizer,
                 max_context_tokens, max_generate_tokens, temperature, do_sample, sample_k, save_path, model=None):
        
        self.model_tokenizer = model_tokenizer
        self.model = model
        self.max_context_tokens = max_context_tokens
        self.max_generate_tokens = max_generate_tokens
        self.temperature = temperature
        self.do_sample = do_sample
        self.sample_k = sample_k
        self.prompt_list = []
        self.output_data_list = []
        self.save_path = save_path
        self.raw_data = raw_data

        self.load_model_format()

    def load_model_format(self, model_format: str = 'vicuna'):
        """support format: alpaca, vicuna"""
        if model_format == 'vicuna':
            sep_stype = Vicuna11SepStyle()
            self.conversation_template = Conversations(sep_style=sep_stype)
        elif model_format == 'alpaca':
            sep_stype = AlpacaSepStyle()
            self.conversation_template = Conversations(sep_style=sep_stype)

    def format_model_input(self, chat_require: ChatRequire):
        """ This function provides utility for formatting the model input.
        
        - truncates the messages by max_context_tokens
        - replaces the character's name if exists
        - pad sending photo or not sending photo prompt
        """
        messages = chat_require.messages
        user_name = "me"
        character_name = chat_require.character_name or "assistant"

        # truncate messages by max_context_tokens
        character_prompt = ""
        instruction_length = 0
        if messages[0].role == "system":
            character_prompt = messages[0].content
            instruction_length = len(self.model_tokenizer(character_prompt).input_ids)
            messages = messages[1:] # remove system_prompt from messages

        context_messages = []
        for message in messages[::-1]:
            instruction_length += len(self.model_tokenizer(message.content).input_ids)
            if instruction_length > self.max_context_tokens:
                break
            # replace character's name if name exists in the message
            if message.name:
                character_name = message.name
            context_messages.append(message)
        context_messages = context_messages[::-1]

        # compose the model_input by conversation format
        self.conversation_template.replace_names(user_name, character_name)
        model_input = self.conversation_template.get_prompt(character_prompt, context_messages)

        return model_input

    def preprocess(self,):
        uid = 0
        pbar = tqdm(self.raw_data)
        for index, data_piece in enumerate(pbar):
            output_data_piece = {}
            output_data_piece['id'] = index
            output_data_piece['character_name'] = data_piece['bot_info']["character_name"]
            output_data_piece["user_name"] = ""
            output_data_piece["conversations"] = []

            dialogue_list = data_piece['dialogue']
            bot_info = data_piece['bot_info']
            system_instruction = bot_info["prompt"]
            system_message = Message(role='system', content=system_instruction)
            message_list = [system_message]
            chat_require = ChatRequire(character_name=bot_info["character_name"], messages=message_list)
            character_name = bot_info["character_name"]
            output_data_piece["conversations"].append({"from": "instruction", "value": system_instruction})

            for content in reversed(dialogue_list):
                if content["speaker_type"] == 1:
                    cur_message = Message(role='assistant', content=content['content'], name=character_name)
                    chat_require.messages.append(cur_message)
                    output_data_piece["conversations"].append({"from": character_name, "value": content['content']})
                else:
                    speaker_name = 'user'
                    cur_message = Message(role='user', content=content['content'], name=speaker_name)
                    chat_require.messages.append(cur_message)
                    model_input_prompt = self.format_model_input(chat_require)
                    self.prompt_list.append(model_input_prompt)
                    output_data_piece["conversations"].append({"from": "Me", "value": content['content']})
                    output_data_piece["conversations"].append({"from": 'sample', "value": uid})
                    uid += 1

            self.output_data_list.append(output_data_piece)

    def rollout(self,):
        outputs_list = []
        sampling_params = SamplingParams(temperature=self.temperature, max_tokens=self.max_generate_tokens)
        for i in range(self.sample_k):
            model_outputs = self.model.generate(self.prompt_list, sampling_params)
            outputs = []
            for output in model_outputs:
                generated_text = output.outputs[0].text
                outputs.append(generated_text)
            outputs_list.append(outputs)
        
        self.outputs_list = outputs_list

    def save_samples(self,):

        pbar = tqdm(self.output_data_list)
        for index, data_piece in enumerate(pbar):
            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            for content in dialogue_list:
                if content['from'] == 'sample':
                    uid = content['value']
                    content['value'] = []
                    for i in range(len(self.outputs_list)):
                        content['value'].append(self.outputs_list[i][uid])

        with open(self.save_path, mode='w') as f:
            json.dump(self.output_data_list, f, indent=4)
        return self.output_data_list

    def save(self, model_outputs):
        pbar = tqdm(self.output_data_list)
        for index, data_piece in enumerate(pbar):
            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            for content in dialogue_list:
                if content['from'] == character_name:
                    uid = content['value']
                    content['value'] = model_outputs[uid]

        with open(self.save_path, mode='w') as f:
            json.dump(self.output_data_list, f, indent=4)





if __name__ == '__main__':
    raw_data = [{"A":0}, {"B": 1}, {"C": 2}]
    pbar = tqdm(raw_data)
    for i, data in enumerate(pbar):
        print(data)