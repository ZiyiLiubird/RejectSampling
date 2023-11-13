import dataclasses
from typing import List, Any, Dict, Union, Tuple
from models.utils import Message

# prompt styles: https://rentry.co/llm_rp_prompts


@dataclasses.dataclass
class BaseSepStyle:
    sep: str = "\n"
    sep2: str = None
    roles=("USER", "ASSISTANT")
    roles_replaced = False
    roles_to_replaced: Tuple = None
    # stops: List[str] = dataclasses.field(default_factory=list)
    

@dataclasses.dataclass
class Vicuna11SepStyle(BaseSepStyle):
    sep: str = " "
    sep2: str = "</s>"
    roles_replaced = False
    roles=("USER", "ASSISTANT")
    roles_to_replaced=("{}", "{}")
    stops=[]
    

@dataclasses.dataclass
class AlpacaSepStyle(BaseSepStyle):
    sep: str = "\n\n"
    sep2: str = ''
    roles_replaced: bool = False
    roles_to_replaced: Tuple = (
        "### Instruction:\n {}", 
        "### Response:\n {}")
    roles: Tuple =(
        "### Instruction:\n ", 
        "### Response:\n ")
    stops = ['###', '### Instruction']

@dataclasses.dataclass
class Conversations:
    """manage prompts and keep history"""
    
    # Seperate style to generate prompt
    sep_style: BaseSepStyle = None

    def get_prompt(self, system_message: str, messages: List[Message]) -> str:
        system_prompt = system_message
        seps = [self.sep_style.sep, self.sep_style.sep2]
        ret = system_prompt + seps[0]
        for i, message in enumerate(messages):
            role, content = message.role, message.content
            if message.role == 'user':
                role = self.sep_style.roles[0]
                sep = seps[0]
            elif message.role == 'assistant':
                role = self.sep_style.roles[1]
                sep = seps[1]
                
            if content:
                ret += role + content + sep
            else:
                ret += role
        
        # add response pre-fix
        ret += self.sep_style.roles[1].strip()
        return ret
    
    def replace_names(self, user_name: str, character_name: str):
        if not self.sep_style.roles_replaced:
            # self.sep_style.roles = (user_name, character_name)
            self.sep_style.stops.append(self.sep_style.roles[0])
        else:
            self.sep_style.roles = (
                self.sep_style.roles_to_replaced[0].format(user_name), 
                self.sep_style.roles_to_replaced[1].format(character_name)
            )
            self.sep_style.stops.append(self.sep_style.roles_to_replaced[0])