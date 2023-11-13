from typing import List
from tqdm import tqdm, trange
import json
import torch
from torch.utils.data.sampler import BatchSampler, SequentialSampler
import numpy as np

from models.utils import ChatRequire
from models.model import get_single_reward_from_model
from models.utils import Message


class Reject:
    def __init__(self, reward_model, reward_tokenizer, max_context_tokens, sample_data, batch_size,
                 result_save_path, sample_save_path, model_device):
        self.reward_model = reward_model
        self.reward_tokenizer = reward_tokenizer
        self.result_save_path = result_save_path
        self.sample_save_path = sample_save_path
        self.max_context_tokens = max_context_tokens
        self.sample_data = sample_data
        self.model_device = model_device
        self.batch_size = batch_size
        self.output_data_list = []
        self.prompt_list = []
        self.answer_list = []
        self.score_list = []

        self.preprocess()

    def format_model_input(self, chat_require: ChatRequire):
        """ This function provides utility for formatting the model input.
        
        - truncates the messages by max_context_tokens
        - replaces the character's name if exists
        - pad sending photo or not sending photo prompt
        """
        messages = chat_require.messages

        # truncate messages by max_context_tokens
        character_prompt = ""
        instruction_length = 0
        if messages[0].role == "system":
            character_prompt = messages[0].content
            instruction_length = len(self.reward_tokenizer(character_prompt).input_ids)
            messages = messages[1:] # remove system_prompt from messages

        context_messages = []
        for message in messages[::-1]:
            instruction_length += len(self.reward_tokenizer(message.content).input_ids)
            if instruction_length > self.max_context_tokens:
                break

            context_messages.append(message)
        context_messages = context_messages[::-1]

        model_input = self.get_prompt(character_prompt, context_messages)

        return model_input

    def get_prompt(self, character_prompt, context_messages:List[Message]):
        prompt = "[Start of Init]: "
        prompt += character_prompt + "[End of Init]"
        for mess in context_messages:
            if mess.role == 'Me':
                prompt += '[User]: ' + mess.content
            else:
                prompt += '[Assistant]: ' + mess.content

    def preprocess(self):
        pbar = tqdm(self.sample_data)
        for index, data_piece in enumerate(pbar):

            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']

            for i, content in enumerate(dialogue_list):
                if content['from'] == character_name:
                    if dialogue_list[i-1]["from"] == "instruction":
                        continue
                    elif dialogue_list[i-1]["from"] == "sample":
                        dialogue_list[i-1]["value"].append(content["value"])

    def batch_preprocess(self,):
        uid = 0
        pbar = tqdm(self.sample_data)
        for index, data_piece in enumerate(pbar):

            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            message_list = []
            chat_require = ChatRequire(character_name='asssistant', messages=message_list)
            output_data_piece = {}
            output_data_piece['id'] = index
            output_data_piece['character_name'] = data_piece["character_name"]
            output_data_piece["user_name"] = data_piece["user_name"]
            output_data_piece["conversations"] = []

            for content in dialogue_list:
                if content["from"] == 'instruction':
                    system_message = Message(role='system', content=content['value'], name='system')
                    chat_require.messages.append(system_message)
                    output_data_piece["conversations"].append({"from": "instruction", "value": content['value']})
                elif content['from'] == character_name:
                    cur_message = Message(role='assistant', content=content['value'], name='assistant')
                    chat_require.messages.append(cur_message)
                    if output_data_piece["conversations"][-1]["from"] == "instruction":
                        output_data_piece["conversations"].append({"from": character_name, "value": content['value']})
                elif content["from"] == "Me":
                    cur_message = Message(role='user', content=content['value'], name='user')
                    chat_require.messages.append(cur_message)
                    model_input_prompt = self.format_model_input(chat_require)
                    self.prompt_list.append(model_input_prompt)
                    output_data_piece["conversations"].append({"from": "Me", "value": content['value']})
                    output_data_piece["conversations"].append({"from": character_name, "value": uid, "score": 0})

                elif content['from'] == 'sample':
                    content['score'] = uid
                    self.answer_list.append(content['value'])
                    uid += 1

            self.output_data_list.append(output_data_piece)

    def single_reject(self):

        pbar = tqdm(self.sample_data)
        for index, data_piece in enumerate(pbar):

            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            message_list = []
            chat_require = ChatRequire(character_name='asssistant', messages=message_list)
            output_data_piece = {}
            output_data_piece['id'] = index
            output_data_piece['character_name'] = data_piece["character_name"]
            output_data_piece["user_name"] = data_piece["user_name"]
            output_data_piece["conversations"] = []

            for content in dialogue_list:
                if content["from"] == 'instruction':
                    system_message = Message(role='system', content=content['value'], name='system')
                    chat_require.messages.append(system_message)
                    output_data_piece["conversations"].append({"from": "instruction", "value": content['value']})
                elif content['from'] == character_name:
                    cur_message = Message(role='assistant', content=content['value'], name='assistant')
                    chat_require.messages.append(cur_message)
                    if output_data_piece["conversations"][-1]["from"] == "instruction":
                        output_data_piece["conversations"].append({"from": character_name, "value": content['value']})
                elif content["from"] == "Me":
                    cur_message = Message(role='user', content=content['value'], name='user')
                    chat_require.messages.append(cur_message)
                    model_input_prompt = self.format_model_input(chat_require)
                    output_data_piece["conversations"].append({"from": "Me", "value": content['value']})
                elif content['from'] == 'sample':
                    best_score = -1e9
                    best_response = ""
                    content['score'] = []
                    for response in content['value']:
                        score = get_single_reward_from_model(model=self.reward_model, tokenizer=self.reward_tokenizer,
                                                             answer=response, message_str=model_input_prompt,
                                                             max_length=self.max_context_tokens, device=self.model_device)
                        score = score.item()
                        print(f"score: {score}")
                        print(f"type: {type(score)}")
                        content['score'].append(score)
                        if score > best_score:
                            best_response = response
                            best_score = score
                    output_data_piece["conversations"].append({"from": character_name, "value": best_response, "score": best_score})

            self.output_data_list.append(output_data_piece)

    @torch.no_grad()
    def get_batch_reward_from_model(self, prompt, answers):
        reward_input_template = 'BEGINNING OF CONVERSATION: USER: {} ASSISTANT:{}'
        inputs = []
        for i in range(len(answers)):
            reward_input = reward_input_template.format(prompt, str(answers[i])) + self.reward_tokenizer.eos_token
            inputs.append(reward_input)

        input_ids = self.reward_tokenizer(inputs, return_tensors="pt", padding=True, truncation=True)['input_ids']
        attention_mask = (input_ids != 0).bool()
        outputs = self.reward_model(input_ids.to(self.model_device), attention_mask.to(self.model_device))
        end_scores = outputs.end_scores.squeeze(dim=-1).cpu().tolist()
        return end_scores

    def batch_reject(self):
        self.batch_preprocess()

        length = len(self.prompt_list)
        for index in trange(length):
            prompt = self.prompt_list[index]
            answers = self.answer_list[index]
            answers = np.array(answers)
            num_batch = len(answers) // self.batch_size
            left = len(answers) % self.batch_size
            scores = []
            for t in range(num_batch):
                sub_answers = answers[t*self.batch_size:(t+1)*self.batch_size]
                end_scores = self.get_batch_reward_from_model(prompt=prompt, answers=sub_answers)
                scores.extend(end_scores)
            if left != 0:
                end_scores = self.get_batch_reward_from_model(prompt=prompt, answers=answers[-left:])
                scores.extend(end_scores)
            self.score_list.append(scores)

        assert len(self.score_list) == len(self.prompt_list)
        self.batch_postprocess()

    def batch_postprocess(self):
        pbar = tqdm(self.output_data_list)
        for index, data_piece in enumerate(pbar):
            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            for t, content in enumerate(dialogue_list):
                if content['from'] == character_name and t != 1:
                    uid = content['value']
                    best_score = -1e9
                    for i in range(len(self.score_list[uid])):
                        if self.score_list[uid][i] > best_score:
                            best_score = self.score_list[uid][i]
                            best_response = self.answer_list[uid][i]
                    content['value'] = best_response
                    content['score'] = best_score

        pbar = tqdm(self.sample_data)
        for index, data_piece in enumerate(pbar):
            dialogue_list = data_piece['conversations']
            character_name = data_piece['character_name']
            for content in dialogue_list:
                if content['from'] == 'sample':
                    uid = content['score']
                    content['score'] = self.score_list[uid]

    def save(self):
        with open(self.result_save_path, mode='w') as f:
            json.dump(self.output_data_list, f, indent=4)
        with open(self.sample_save_path, mode='w') as f:
            json.dump(self.sample_data, f, indent=4)
        print(f"Successful Saved !")