import logging

import fastapi

from research_tool_rag.configs import config
from research_tool_rag.rag.pipeline import RAGPipeline
from research_tool_rag.state.state_schema import OutputState

app = fastapi.FastAPI(
    title="Research Tool",
    description="API service to research tool",
    version="0.1.0",
)

config.use_config("online")
pipeline = RAGPipeline()


@app.get("/")
async def read_root():
    """Basic health check endpoint."""
    logging.info("Root endpoint '/' accessed (health check).")
    return {"status": "ok", "message": "Document Change Analyzer API is running."}


@app.post("/rag", response_model=OutputState)
def rag_query(question: str):

    if not question:
        question = (
            "Explain the penalties for violating agricultural market regulations in New York."
        )
    result = pipeline.run(question=question)
    return result
