# from research_tool_rag.rag.ingest_data import Hierarchy
import logging

from research_tool_rag.configs import config
from research_tool_rag.rag.pipeline import RAGPipeline
from research_tool_rag.utils.utils import setup_logging

logger = logging.getLogger(__name__)

setup_logging("ingest_data", stream_handler=True)

config.use_config("online")
pipeline = RAGPipeline()
question = input("Hi! How can I help you today?")
if not question:
    question = "Explain the penalties for violating agricultural market regulations in New York."
result = pipeline.run(question=question)
print(result)
