import abc
from typing import Optional, Union, List
import torch
from utils import Message, ChatRequire
from prepost import Vicuna11SepStyle, AlpacaSepStyle, Conversations 




class ChatModel():
    __metaclass__ = abc.ABCMeta

    @abc.abstractclassmethod
    def format_model_input(self, chat_require: ChatRequire):
        raise NotImplementedError

    @abc.abstractclassmethod
    def generate_response(
        self, 
        chat_require: ChatRequire, 
    ):
        raise NotImplementedError

    @abc.abstractclassmethod
    def format_model_output(self):
        raise NotImplementedError

    @abc.abstractclassmethod
    def load_model_and_tokenizer(self):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def model_generate_stream(self):
        raise NotImplementedError
    
    @abc.abstractclassmethod
    def model_generate_nonstream(self):
        raise NotImplementedError


class EnglishChatModel(ChatModel):
    def __init__(self, model_config: dict, prompt_config: dict, device: torch.device) -> None:
        super().__init__()
        self.device = device
        self.prompt_config = prompt_config
        self.model_path = model_config['ModelPath']
        self.max_context_tokens = model_config['MaxContextTokens']
        self.load_model_format('vicuna')
        self.load_model_and_tokenizer()
        
    def load_model_format(self, model_format: str):
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
        
        return model_input, character_name

    def generate_response(
        self, 
        chat_require: ChatRequire,
    ):

        # format model input
        model_input, character_name = self.format_model_input(chat_require)

        # request requires non-streaming process
        chat_choices, usage_list = [], []
        for index in range(chat_require.n):
            model_output, usage = self.model_generate_nonstream(
                model_input=model_input,
                max_out_length=chat_require.max_tokens,
                top_p=chat_require.top_p,
                temperature=chat_require.temperature
            )
            
