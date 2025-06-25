from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel
from typing_extensions import List


class State(MessagesState):
    question: str
    context: List[Document]
    answer: str
    source: List[str]


class InputState(MessagesState):
    question: str
    context: str


class OutputState(BaseModel):
    answer: str
    sources: List[str]


class OverallState(MessagesState):
    question: str
    context: List[Document]
    answer: str
    source: List[str]
