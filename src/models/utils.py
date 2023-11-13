from typing import Optional, Union, List
from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str = ""
    content: str = ""
    name: str = ""


class ChatRequire(BaseModel):
    character_name: Union[str, None]
    messages: List[Message]
    temperature: Optional[float]
    top_p: Optional[float]
    n: Optional[int]=1
    stream: Optional[bool]
    stop: Optional[List[str]]
    max_tokens: Optional[int]
    presence_penalty: Optional[float]
    frequency_penalty: Optional[float]
