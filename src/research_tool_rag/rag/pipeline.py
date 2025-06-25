from langchain import hub
from langgraph.graph import START, StateGraph

from research_tool_rag.configs import config
from research_tool_rag.db_store.qdrant import QdrantDB
from research_tool_rag.state.state_schema import InputState, OutputState, OverallState


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

    def retrieve(self, state: InputState) -> OverallState:
        retrieved_docs = self.vector_store.similarity_search(state["question"])
        return {"question": state["question"], "context": retrieved_docs}

    def generate(self, state: OverallState) -> OutputState:
        source = [doc.metadata["hierarchical_name"] for doc in state["context"]]
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({"question": state["question"], "context": docs_content})
        response = self.llm.invoke(messages)
        return {"answer": response.content, "sources": source}

    def _build_graph(self):
        graph_builder = StateGraph(InputState, output=OutputState)
        graph_builder.add_node("retrieve", self.retrieve)
        graph_builder.add_node("generate", self.generate)
        graph_builder.add_edge(START, "retrieve")
        graph_builder.add_edge("retrieve", "generate")
        graph = graph_builder.compile()
        return graph

    def run(self, question: str):
        response = self.graph.invoke({"question": question})
        return response
