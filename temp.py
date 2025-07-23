import logging
from typing import Literal, Optional, Annotated, List

from langchain.tools import Tool
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.messages import trim_messages, HumanMessage, SystemMessage, ToolMessage, ChatMessage
from langgraph.types import Command, interrupt
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, PromptTemplate
from langgraph.checkpoint.memory import MemorySaver
from research_tool_rag.configs import config
from research_tool_rag.rag.pipeline import RAGPipeline
from research_tool_rag.utils.utils import setup_logging

# Setup logging
logger = logging.getLogger(__name__)
setup_logging("ingest_data", stream_handler=True)

# Initialize config and pipeline
config.use_config("online")
pipeline = RAGPipeline()

# Literal types for US states and law type
USStateAbbrev = Literal[
    "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN",
    "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV",
    "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN",
    "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"
]
LawType = Literal["laws", "regulations"]

# Define the graph state
class ExtractState(MessagesState):
    query: str
    state_code: Optional[str]
    law_type: Optional[str]
    expanded_query: Optional[str]
    correction: str
    tool_attempts: int = 0
    context: str
    answer: str
    sources: List['str']
    tool_attempts: int = 0

# Tool to update state with US state and law type
@tool
def get_us_state_lawtype(
    tool_call_id: Annotated[str, InjectedToolCallId],
    state_abbrev: USStateAbbrev,
    law_type: LawType = ""
) -> Command:
    """Update state with US state abbreviation and optional law type."""
    return Command(update={
        "state_code": state_abbrev,
        "law_type": law_type,
        "messages": [ToolMessage("Updated state_code and law_type", tool_call_id=tool_call_id)]
    })

# Bind tools to LLM
llm_with_tools = pipeline.llm.bind_tools([get_us_state_lawtype])

# Assistant node to extract state and law type
def assistant(state: ExtractState):
    messages = trim_messages(
        state["messages"], max_tokens=100000, strategy="last", token_counter=pipeline.llm, allow_partial=False
    )
    response = llm_with_tools.invoke(
        [SystemMessage(content="Extract state code and law type from the following messages.")] + messages[-1:]
    )
    return {"messages": [response], 'query':messages[0].content}

# Query builder node to expand query
def query_builder(state: ExtractState):
    prompt_template = ChatPromptTemplate(
        input_variables=["query"],
        messages=[
            HumanMessagePromptTemplate(
                prompt=PromptTemplate(
                    input_variables=["query"],
                    template="Expand or refine the following legal query for clarity and precision:\nQuery: {query}\nExpanded query:"
                )
            )
        ]
    )
    message = prompt_template.invoke({"query": state["query"]})
    response = llm_with_tools.invoke(message)
    return {"messages": [response], "expanded_query": response.content}

# Conditional edge selector
def state_checker_router(state: ExtractState) -> Command[Literal['query_builder','gather_info','assistant']]:
    if state.get("state_code"):
        goto="query_builder"
        tool_attmpt = 0
    elif state.get("tool_attempts", 0) >= 1:
        tool_attmpt = 0
        goto='gather_info'
    else:
        tool_attmpt = state.get("tool_attempts", 0)+1
        goto="assistant"

    return Command(update={
            "tool_attempts": tool_attmpt,
            # "messages": [ChatMessage(content="Updated tool_attempt number and going to gather info",role='bot')]
            },
            goto=goto)

# Gather info node (interrupt-based)
def gather_info(state: ExtractState)->  ExtractState:
    value =  interrupt("what is the state and laws you are interested in")
    return value

# # Wrapper node for tool with retry counter
# def tool_with_retry(tool_call_id: Annotated[str, InjectedToolCallId], state: ExtractState):
#     breakpoint()
#     state["tool_attempts"] = state.get("tool_attempts", 0) + 1
#     Command(update={
#         "tool_attempts": state.get("tool_attempts", 0) + 1,
#         "messages": [ToolMessage("Updated tool_attempt number", tool_call_id=tool_call_id)]
#     }
#     goto=)
#     return ToolNode([get_us_state_lawtype]).invoke(state)

# Retrieval node using RAGPipeline
def retrieve(state: ExtractState):
    retrieved = pipeline.retrieve({"question": state["query"],"state":state['state_code']})
    return {"context": retrieved["context"]}

# Generation node using RAGPipeline
def generate(state: ExtractState):
    overall_state = {
        "question": state["query"], # state["expanded_query"] or 
        "context": state["context"]
    }
    generated = pipeline.generate(overall_state)
    return {"answer": generated["answer"], "sources": generated["sources"]}

# Build LangGraph
builder = StateGraph(ExtractState)
builder.add_node("assistant", assistant)
builder.add_node("gather_info", gather_info)
builder.add_node("query_builder", query_builder)
builder.add_node("state_checker_router",state_checker_router)
builder.add_node("tools", ToolNode([get_us_state_lawtype]))
builder.add_node("retrieve", retrieve)
builder.add_node("generate", generate)

builder.add_edge(START, "assistant")
builder.add_edge("assistant", "tools")
builder.add_edge("tools", "state_checker_router")
builder.add_edge("gather_info", "tools")
builder.add_edge("query_builder", "retrieve")
builder.add_edge("retrieve", "generate")
builder.add_edge("generate", END)
# checkpointer = MemorySaver()
# rag_graph = builder.compile(checkpointer=checkpointer,debug=True)
rag_graph = builder.compile()
# display(Image(rag_graph.get_graph().draw_mermaid_png()))
if __name__ == "__main__":
    messages = [
        HumanMessage(content="Explain the penalties for violating agricultural market regulations in New York.")# 
    ]
    config_obj = {"configurable": {"thread_id": "test-thread"}}  # Optional but good for interrupt tracking

    # Initial run
    result = rag_graph.invoke({"messages": messages}, config=config_obj)

    while "__interrupt__" in result:
        interrupt_payload = result["__interrupt__"]
        print(f"\n🚨 Interrupt triggered: {interrupt_payload}")

        # Simulate user providing the missing information via CLI
        user_state = input(f"\n{interrupt_payload[0].value}\nProvide your input: ")
        # Resume the graph with user's input
        result = rag_graph.invoke(
            Command(resume={"messages": [HumanMessage(content=user_state)]}),
            config=config_obj
        )

    # Final result
    for m in result.get("messages", []):
        m.pretty_print()
    print("\nFinal Answer:", result.get("answer"))
    print("Sources:", result.get("sources"))
