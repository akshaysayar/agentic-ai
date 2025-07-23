from langchain import hub
from langgraph.graph import START, StateGraph
from qdrant_client import models
from research_tool_rag.configs import config
from research_tool_rag.db_store.qdrant import QdrantDB
from research_tool_rag.state.state_schema import InputState, OutputState, OverallState
from typing import Literal
import json, re

class RAGPipeline:
    def __init__(self):
        self.db = QdrantDB()
        self.vector_store = self.db.vector_store
        self.prompt = hub.pull("rlm/rag-prompt")
        self.prompt.messages[
            0
        ].prompt.template = """You are a helpful and accurate legal assistant specialized in answering questions based on US Laws. Use only the information provided in the retrieved context to answer the question. If the answer cannot be found in the context, clearly state that you don't know. Keep your answer concise, legally precise, and limited to a maximum of three sentences.
Question: {question}
Context: {context}
Answer:"""
        self.llm = config.llm
        self.graph = self._build_graph()

        # US State Abbreviations as Literal Type
        self.USStateAbbrev = Literal["AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD", "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ", "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC", "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY"]


        self.LawType = Literal["laws", "regulations"]

    def retrieve(self, state: InputState) -> OverallState:
        # retriever= self.vector_store.as_retriever(search_kwargs={'k':4,'filter': {'state':state.get("state_code",'ny').lower()}})
        # retrieved_docs = retriever.invoke(state['question'])
        retrieved_docs = self.vector_store.similarity_search(query=state["question"])
                                                # filter={"state":state.get('state_code','ny').lower()})
        return {"question": state["question"], "context": retrieved_docs}

    # def _build_graph(self):
        # graph_builder = StateGraph(InputState, output=OutputState)
        # graph_builder.add_node("retrieve", self.retrieve)
        # graph_builder.add_node("generate", self.generate)
        # graph_builder.add_edge(START, "retrieve")
        # graph_builder.add_edge("retrieve", "generate")
        # graph = graph_builder.compile()
        # return graph

    def run(self, question: str, thread: int = 1):
        config = {"configurable": {"thread_id": str(thread)}}
        response = self.graph.invoke({"question": question}, config=config)
        # breakpoint()
        return response

    def _build_graph(self):
        graph_builder = StateGraph(InputState, output=OutputState)

        graph_builder.add_node("classify_query", self.classify_query)
        graph_builder.add_node("rewrite_query", self.hyde_generate)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate_from_context_with_suggestions", self.generate_from_context_with_suggestions)
        graph_builder.add_node("generate_direct_with_suggestions", self.generate_direct_with_suggestions)

        graph_builder.add_conditional_edges(
            START,
            self.classify_query,
            ['retrieve','generate_direct_with_suggestions','rewrite_query']
        )

        graph_builder.add_edge("rewrite_query", "retrieve")
        graph_builder.add_edge("retrieve", "generate_from_context_with_suggestions")

        graph = graph_builder.compile()
        return graph


    def llm_invoke(self, prompt):
        response = self.llm.invoke(prompt)
        return response.content if hasattr(response, "content") else response

    def classify_query(self, state: InputState) -> Literal['generate_direct_with_suggestions','retrieve','rewrite_query']:
        prompt = f"""
You are an intelligent reasoning assistant for a research chatbot.

Given a user's question, your task is to decide how the system should process it. Choose ONLY one of the following actions:

- "general" → The question is conversational, a greeting, or general knowledge. It can be answered directly by the chatbot without searching external documents.

- "good_for_retrieval" → The question is clear and specific enough to retrieve relevant documents from the knowledge base before answering.

- "needs_hyde" → The question is vague, incomplete, or not well-formed for document retrieval. The system should generate a hypothetical document to reformulate the query before retrieval.

Question:
{state['question']}

Respond ONLY with one of: general, good_for_retrieval, needs_hyde

Action:
"""
        classify_keys = {"general":"generate_direct_with_suggestions","good_for_retrieval":"retrieve","needs_hyde":"rewrite_query"}
        classification = self.llm_invoke(prompt).strip().lower()
        # breakpoint()
        if classification in {"generate", "good_for_retrieval","needs_hyde"}:
            next_step = classify_keys.get(classification,"generate_direct_with_suggestions")
            return next_step
        return "generate_direct_with_suggestions"
    

    def hyde_generate(self, state: OverallState)-> OverallState:
        hyde_prompt = f"""
You are a helpful assistant generating a hypothetical document or enriched query to improve information retrieval.
Given the following user question, rewrite it with additional detail to make retrieval more effective. The retrieval used cosine similarity with the stored documents.
The stored documents are about laws and regulation in US for all states and federal too.

User Question:
{state['question']}

Enriched Query:
"""
        enriched_query = self.llm_invoke(hyde_prompt).strip()
        # enriched_query = self.llm.invoke(hyde_prompt)
        return {'question':enriched_query or self['question']}

    
    def generate_from_context_with_suggestions(self, state: OverallState)-> OutputState:
        source = [doc.metadata["hierarchical_name"] for doc in state["context"]]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        prompt = f"""
You are an intelligent assistant for a research chatbot. 

For every request, respond ONLY with a valid JSON object matching the following format:

{{
  "answer": "<Your clear, helpful answer to the user's question, based on the provided documents. If the documents do not contain relevant information, say so politely.>",
  "suggested_prompts": ["<First suggested follow-up prompt>", "<Second prompt>", "<Third prompt>"]
}}

The "suggested_prompts" list can contain 1 to 3 concise, relevant follow-up questions or topics the user might ask next to continue the conversation.

Do not include any text outside the JSON block. No explanations, no extra formatting — only valid JSON.

### User's Question:
{state['question']}

### Retrieved Documents:
{docs_content}

Respond with JSON format only:
"""

        llm_output = self.llm_invoke(prompt)
        
        try:
            llm_output = json.loads(re.sub(r"^```json\s*|```$", "", llm_output.strip(), flags=re.IGNORECASE).strip())
            answer = llm_output['answer']
            suggested_prompts = llm_output['suggested_prompts']
        except (json.JSONDecodeError, ValueError) as e:
            answer = "⚠️ There was an error processing the response."
            suggested_prompts = []

        
        return {'sources':source, 'suggested_prompts':suggested_prompts, 'answer':answer}
    
    def generate_direct_with_suggestions(self, state: OverallState)-> OutputState:
        prompt = f"""
You are a helpful chatbot that answers general, conversational, or knowledge-based questions.

For every request, respond ONLY with a valid JSON object matching the following format:

{{
  "answer": "<Your helpful, friendly answer to the user's question>",
  "suggested_prompts": ["<First follow-up prompt>", "<Second follow-up prompt>", "<Third follow-up prompt>"]
}}

The "suggested_prompts" list can contain 1 to 3 concise, relevant follow-up questions or topics to keep the conversation going.
Remember that you are a legal assistant and have knowledge of US federal and states laws. Suggest follow up prompts by keeping this in mind.
ALWAYS suggest prompts that users are likely to ask next.

Do not include any text outside the JSON block. No commentary, no extra explanations — only valid JSON.

User's Question:
{state['question']}

Respond with JSON:
"""
        llm_output = self.llm_invoke(prompt)
        try:
            llm_output = json.loads(re.sub(r"^```json\s*|```$", "", llm_output, flags=re.IGNORECASE).strip())
            answer = llm_output['answer']
            suggested_prompts = llm_output['suggested_prompts']
        except (json.JSONDecodeError, ValueError) as e:
            answer = "⚠️ There was an error processing the response."
            suggested_prompts = []
        
        return {'suggested_prompts':suggested_prompts, 'answer':answer}
