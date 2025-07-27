import getpass
import os

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


model_db_config = {
    "llm": init_chat_model("gemini-2.0-flash", model_provider="google_genai"),
    "embeddings": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
    "db_url": "localhost",
    "db_port": 6333,
    "collection_name": "US_LAWS",
    "collection_distance_metric": "Cosine",
}
