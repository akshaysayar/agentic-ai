import logging
from typing import Literal, Optional

from langchain.tools import Tool
from typing import Annotated
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
# from langchain_community.chat_models import BaseChatModel
from langchain_core.messages import trim_messages
from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.types import Command, interrupt
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain import hub
from research_tool_rag.configs import config
from research_tool_rag.rag.pipeline import RAGPipeline
from research_tool_rag.utils.utils import setup_logging
from langchain_core.tools import tool, InjectedToolCallId
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate


logger = logging.getLogger(__name__)

setup_logging("ingest_data", stream_handler=True)

# Initialize Config and Pipeline
config.use_config("online")
pipeline = RAGPipeline()



# US State Abbreviations as Literal Type
USStateAbbrev = Literal["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


LawType = Literal["laws", "regulations"]


# Category = Annotated[
#     Literal['laws', 'regulations', 'not_found'],
#     BeforeValidator(lambda v: v.lower())
# ]
class ExtractState(MessagesState):
    query: str
    state_code: Optional[str]
    law_type: Optional[str]
    expanded_query: Optional[str]
    correction: str


@tool
def get_law_type(law_type: LawType) -> Literal["laws", "regs", ""]:
    """Extract and return the law type from the question provided by the LLM.
    This law_type will then be passed as metadata to narrow down the query.
    If the law_type is not clear, we return ''
    """
    if law_type.lower() == "laws":
        return "LAWS"
    else:
        return "REGS"


# Tool for extracting US state from LLM output
@tool
def get_us_state(state_abbrev: USStateAbbrev) -> USStateAbbrev:
    """Extract and return the US state abbreviation from the question provided by the LLM.
    This state_abbrev will then be passed as metadata to narrow down the query."""
    return state_abbrev


get_law_type_tool = Tool(
    name="Get Law Type",
    func=get_law_type,
    description="Extracts the law type and passes returns them simply",
)
get_us_state_tool = Tool(
    name="Get State Code",
    func=get_us_state,
    description="Extracts the State Code and passes returns them simply",
)

@tool
def get_us_state_lawtype(
    tool_call_id: Annotated[str, InjectedToolCallId], state_abbrev: USStateAbbrev, law_type: LawType = ""
) -> Command:
    """Extract the US state abbreviation from the question provided by the LLM.
    This state_abbrev will then be set to State['state_code'].
    Extract the law type from the question provided by the LLM.
    This law_type will then be set as State['law_type'] This is optional var
    """
    return Command(update={"state_code":state_abbrev,"law_type": law_type, "messages":[ToolMessage(
                "Successfully updated state_code and law_type",
                tool_call_id=tool_call_id
            )]})


# Bind the tool to your LLM
llm_with_tools = pipeline.llm.bind_tools([get_us_state_lawtype])
sys_msg = SystemMessage(
    content="You are a helpful and accurate legal assistant specialized in answering questions based on US Laws. Use only the information provided in the retrieved context to answer the question. If the answer cannot be found in the context, clearly state that you don't know. Keep your answer concise, legally precise, and limited to a maximum of three sentences. "
)


def assistant(state: ExtractState):
    messages = trim_messages(
            state["messages"],
            max_tokens=100000,
            strategy="last",
            token_counter=pipeline.llm,
            allow_partial=False,
        )
    return {
        "messages": [
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=f"You are state code and state law type extractor from the messsages: "
                    )
                ]
                + messages[-1:]
            )
        ]
    }


def query_builder(state: ExtractState):
    prompt_template = ChatPromptTemplate(
        input_variables=["query"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["query"],
                    template="You are a helpful legal assistant. Expand or refine the following user query to make it more detailed and legally precise. If you cannot then leave it as it is.\nQuery: {query}\nExpanded query:"
                )
            )
        ]
    )
    message = prompt_template.invoke({"query": state["query"]})
    response = llm_with_tools.invoke(message)
    return {"messages": [response]}


def meta_or_query_builder(state: ExtractState) -> Literal["query_builder", "gather_info"]:
    if 'state_code' in state.keys() and state['state_code']:
        return "query_builder"
    return "gather_info"


def gather_info(state: ExtractState):
    value = interrupt( 
        {
            "text_to_revise": state["query"] 
        }
    )
    print("inside gather_info")
    user_value = input(state["messages"][-1].content+"\n Type which state you would like to search in ->  ")
    return {"messages": [HumanMessage(content=f"{state['messages'][-1].content} for {user_value}")]}


get_us_state_lawtype_tool = Tool(
    name="get_us_state_lawtype",
    func=get_us_state_lawtype,
    description="Extracts the State Code and laws and apply to state",
)

builder = StateGraph(ExtractState)

# Define nodes: these do the work
builder.add_node("assistant", assistant)
builder.add_node("gather_info", gather_info)
builder.add_node("query_builder", query_builder)
builder.add_node("tools", ToolNode([get_us_state_lawtype]))

# Define edges: these determine how the control flow moves
builder.add_edge(START, "assistant")
builder.add_edge("assistant", "tools")
builder.add_conditional_edges("tools", meta_or_query_builder)
builder.add_edge("gather_info", "assistant")
builder.add_edge("query_builder", END)
react_graph = builder.compile()

messages = [
    HumanMessage(
        content="Explain the penalties for violating agricultural market" # regulations in new york
    )
]  # regulations in new york
resp = react_graph.invoke({"messages": messages, "query": messages[0].content})
mm = resp["messages"]
for m in mm:
    m.pretty_print()
