from fastapi import FastAPI
from pydantic import BaseModel
from research_tool_rag.configs import config
from research_tool_rag.rag.pipeline import RAGPipeline

app = FastAPI(
    title="Research Tool",
    description="API service to research tool",
    version="0.1.0",
)

# Init configs and pipeline
config.use_config("online")
pipeline = RAGPipeline()

# Chat sessions if you plan to extend with memory
chat_sessions = {}

# Health check
# @app.get("/")
# async def read_root():
#     logging.info("Root endpoint '/' accessed (health check).")
#     return {"status": "ok", "message": "Research Tool API is running."}

# Request model for RAG endpoint
class RAGRequest(BaseModel):
    query: str
    chat_history: list[dict] = []

# RAG endpoint compatible with Streamlit
# @app.post("/rag")#, response_model=OutputState)
# def rag_query(request: RAGRequest):
#     question = request.query or "Explain the penalties for violating agricultural market regulations in New York."
#     print(question)
#     # You can optionally pass chat_history to your pipeline if supported
#     result = pipeline.run(question=question)
#     # result = {'answer':'this is generated msg','role':'bot','content':'message content',"suggested_prompts": ['whats up','nothing much']}
#     print(result['answer'])
#     return result
question = "Explain the penalties for violating agricultural market regulations in New York."
question = "HI"
result = pipeline.run(question=question)
print(result)