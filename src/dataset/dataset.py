from typing import Dict, Optional, Sequence
import torch
from torch.utils.data import Dataset



class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data, labels):
        super(SupervisedDataset, self).__init__()


        self.tokenizer = tokenizer
        self.device = device
        self.max_context_tokens = max_context_tokens
        self.output_data_list = []
        self.train_dataset = []

        self.load_model_format()
        self.preprocess(raw_data)


    def __len__(self):
        return len(self.train_dataset)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return self.train_dataset[i]

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
            instruction_length = len(self.tokenizer(character_prompt).input_ids)
            messages = messages[1:] # remove system_prompt from messages
        
        context_messages = []
        for message in messages[::-1]:
            instruction_length += len(self.tokenizer(message.content).input_ids)
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

    def preprocess(self, raw_data):
        uid = 0
        for index, data_piece in enumerate(raw_data):
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

            output_data_piece["conversations"].append({"from": "instruction", "value": system_message})


            for content in dialogue_list:
                if content["speaker_type"] == 1:
                    speaker_name = bot_info["character_name"]
                    cur_message = Message(role='assistant', content=content['content'], name=speaker_name)
                    chat_require.messages.append(cur_message)

                    output_data_piece["conversations"].append({"from": speaker_name, "value": content['content']})

                else:
                    speaker_name = 'user'
                    cur_message = Message(role='user', content=content['content'], name=speaker_name)
                    chat_require.messages.append(cur_message)
                    uid += 1
                    output_data_piece["conversations"].append({"from": "Me", "value": uid})
                    model_input_prompt = self.format_model_input(chat_require)
                    # inputs = self.tokenizer([model_input_prompt], return_tensors="pt").to(self.device)
                    # prompt_ids = inputs['input_ids']
                    # prompt_ids = self.tokenizer.encode(model_input_prompt)
                    # print(f"shape: {len(prompt_ids)}")
                    self.train_dataset.append({'prompt': model_input_prompt, "uid": uid})

            self.output_data_list.append(output_data_piece)
