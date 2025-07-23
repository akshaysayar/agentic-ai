from langchain_core.documents import Document
from langgraph.graph import MessagesState
from pydantic import BaseModel
from typing_extensions import List
from pydantic import BaseModel, Field

class State(MessagesState):
    question: str
    context: List[Document]
    answer: str
    source: List[str]


class InputState(MessagesState):
    question: str


class OutputState(BaseModel):
    answer: str =Field(..., description="The chatbot's helpful response to the user's question.")
    sources: List[str]
    suggested_prompts: List[str] = Field(
        default_factory=list, description="1 to 3 concise follow-up prompts for the user."
    )


class OverallState(MessagesState):
    question: str
    context: List[Document]
    answer: str
    source: List[str]
