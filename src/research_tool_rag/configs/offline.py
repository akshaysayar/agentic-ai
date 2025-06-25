# from langchain_community.chat_models import ChatOllama
# from langchain_community.embeddings import OllamaEmbeddings
from langchain_ollama import OllamaEmbeddings, OllamaLLM

model_db_config = {
    "llm": OllamaLLM(model="mistral"),
    "embeddings": OllamaEmbeddings(model="mistral"),
    "db_url": "localhost",
    "db_port": 6333,
    "collection_name": "US_LAWS_offline",
    "collection_distance_metric": "Cosine",
}
