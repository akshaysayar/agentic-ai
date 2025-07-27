import json
import re
from typing import Literal

from langchain import hub
from langgraph.graph import START, StateGraph

from research_tool_rag.configs import config
from research_tool_rag.db_store.qdrant import QdrantDB
from research_tool_rag.state.state_schema import InputState, OutputState, OverallState


class ProfileAgent:
    def __init__(self):
        self.db = QdrantDB()
        self.vector_store = self.db.vector_store
        self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt.messages[0].prompt.template = (
            """You are a helpful and accurate assistant specialized in answering questions based on the index Akshay's profile. Use only the information provided in the retrieved context to answer the question. If the answer cannot be found in the context, clearly state that you don't know. Keep your answer concise and limited to a maximum of three sentences.\nQuestion: {question}\nContext: {context}\nAnswer:"""
        )
        self.llm = config.llm
        self.graph = self._build_graph()

        # US State Abbreviations as Literal Type (if needed for profile context)
        self.USStateAbbrev = Literal[
            "AL",
            "AK",
            "AZ",
            "AR",
            "CA",
            "CO",
            "CT",
            "DE",
            "FL",
            "GA",
            "HI",
            "ID",
            "IL",
            "IN",
            "IA",
            "KS",
            "KY",
            "LA",
            "ME",
            "MD",
            "MA",
            "MI",
            "MN",
            "MS",
            "MO",
            "MT",
            "NE",
            "NV",
            "NH",
            "NJ",
            "NM",
            "NY",
            "NC",
            "ND",
            "OH",
            "OK",
            "OR",
            "PA",
            "RI",
            "SC",
            "SD",
            "TN",
            "TX",
            "UT",
            "VT",
            "VA",
            "WA",
            "WV",
            "WI",
            "WY",
        ]
        self.LawType = Literal["laws", "regulations"]

    def retrieve(self, state: InputState) -> OverallState:
        retrieved_docs = self.vector_store.similarity_search(query=state["question"])
        breakpoint()  # For debugging, remove in production
        return {"question": state["question"], "context": retrieved_docs}

    def run(self, question: str, thread: int = 1):
        config = {"configurable": {"thread_id": str(thread)}}
        response = self.graph.invoke({"question": question}, config=config)
        return response

    def _build_graph(self):
        graph_builder = StateGraph(InputState, output=OutputState)
        graph_builder.add_node("classify_query", self.classify_query)
        graph_builder.add_node("rewrite_query", self.hyde_generate)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node(
            "generate_from_context_with_suggestions", self.generate_from_context_with_suggestions
        )
        graph_builder.add_node(
            "generate_direct_with_suggestions", self.generate_direct_with_suggestions
        )
        # graph_builder.add_conditional_edges(
        #     START,
        #     self.classify_query,
        #     ["retrieve", "generate_direct_with_suggestions", "rewrite_query"]
        # )
        # graph_builder.add_edge("rewrite_query", "retrieve")
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate_from_context_with_suggestions")
        graph = graph_builder.compile()
        return graph

    def llm_invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def classify_query(
        self, state: InputState
    ) -> Literal["generate_direct_with_suggestions", "retrieve", "rewrite_query"]:
        prompt = f"""
You are an intelligent reasoning assistant for a profile Q&A chatbot.\n\nGiven a user's question, your task is to decide how the system should process it. Choose ONLY one of the following actions:\n\n- \"general\" → The question is conversational, a greeting, or general knowledge. It can be answered directly by the chatbot without searching the profile.\n\n- \"good_for_retrieval\" → The question is clear and specific enough to retrieve relevant information from the profile before answering.\n\n- \"needs_hyde\" → The question is vague, incomplete, or not well-formed for retrieval. The system should generate a hypothetical document to reformulate the query before retrieval.\n\nQuestion:\n{state['question']}\n\nRespond ONLY with one of: general, good_for_retrieval, needs_hyde\n\nAction:\n"""
        classify_keys = {
            "general": "generate_direct_with_suggestions",
            "good_for_retrieval": "retrieve",
            "needs_hyde": "rewrite_query",
        }
        classification = self.llm_invoke(prompt).strip().lower()
        if classification in {"general", "good_for_retrieval", "needs_hyde"}:
            next_step = classify_keys.get(classification, "generate_direct_with_suggestions")
            return next_step
        return "generate_direct_with_suggestions"

    def hyde_generate(self, state: OverallState) -> OverallState:
        hyde_prompt = f"""
You are a helpful assistant generating a hypothetical document or enriched query to improve information retrieval from a profile.\nGiven the following user question, rewrite it with additional detail to make retrieval more effective.\n\nUser Question:\n{state['question']}\n\nEnriched Query:\n"""
        enriched_query = self.llm_invoke(hyde_prompt).strip()
        return {"question": enriched_query or state["question"]}

    def generate_from_context_with_suggestions(self, state: OverallState) -> OutputState:
        source = [doc.metadata.get("hierarchical_name", "") for doc in state["context"]]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = f"""
You are an intelligent assistant for a profile Q&A chatbot.\n\nFor every request, respond ONLY with a valid JSON object matching the following format:\n\n{{\n  "answer": "<Your clear, helpful answer to the user's question, based on the provided profile context. If the profile does not contain relevant information, say so politely.>",\n  "suggested_prompts": ["<First suggested follow-up prompt>", "<Second prompt>", "<Third prompt>"]\n}}\n\nThe \"suggested_prompts\" list can contain 1 to 3 concise, relevant follow-up questions or topics the user might ask next to continue the conversation.\n\nDo not include any text outside the JSON block. No explanations, no extra formatting — only valid JSON.\n\n### User's Question:\n{state['question']}\n\n### Retrieved Profile Context:\n{docs_content}\n\nRespond with JSON format only:\n"""
        llm_output = self.llm_invoke(prompt)
        try:
            llm_output = json.loads(
                re.sub(r"^```json\s*|```$", "", llm_output.strip(), flags=re.IGNORECASE).strip()
            )
            answer = llm_output["answer"]
            suggested_prompts = llm_output["suggested_prompts"]
        except (json.JSONDecodeError, ValueError):
            answer = "⚠️ There was an error processing the response."
            suggested_prompts = []
        return {"sources": source, "suggested_prompts": suggested_prompts, "answer": answer}

    def generate_direct_with_suggestions(self, state: OverallState) -> OutputState:
        prompt = f"""
You are a helpful chatbot that answers general, conversational, or knowledge-based questions about the profile.\n\nFor every request, respond ONLY with a valid JSON object matching the following format:\n\n{{\n  "answer": "<Your helpful, friendly answer to the user's question>",\n  "suggested_prompts": ["<First follow-up prompt>", "<Second follow-up prompt>", "<Third follow-up prompt>"]\n}}\n\nThe \"suggested_prompts\" list can contain 1 to 3 concise, relevant follow-up questions or topics to keep the conversation going.\nALWAYS suggest prompts that users are likely to ask next.\n\nDo not include any text outside the JSON block. No commentary, no extra explanations — only valid JSON.\n\nUser's Question:\n{state['question']}\n\nRespond with JSON:\n"""
        llm_output = self.llm_invoke(prompt)
        try:
            llm_output = json.loads(
                re.sub(r"^```json\s*|```$", "", llm_output, flags=re.IGNORECASE).strip()
            )
            answer = llm_output["answer"]
            suggested_prompts = llm_output["suggested_prompts"]
        except (json.JSONDecodeError, ValueError):
            answer = "⚠️ There was an error processing the response."
            suggested_prompts = []
        return {"suggested_prompts": suggested_prompts, "answer": answer}

    # Placeholder for langgraph flow

    def run_agent(self, question: str):
        response = self.graph.invoke({"question": question}, config=config)
        # breakpoint()
        return response


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run agent on profile Qdrant DB.")
    parser.add_argument("--query", type=str, required=True, help="Query for the agent")
    args = parser.parse_args()
    print(run_agent(args.query))
