import getpass
import logging
import os
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# from src.research_tool_rag.configs.defaults import defaults


if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

config_mode = {
    "online": {
        "llm": init_chat_model("gemini-2.0-flash", model_provider="google_genai"),
        "embeddings": GoogleGenerativeAIEmbeddings(model="models/embedding-001"),
        "db_url": "localhost",
        "db_port": 6333,
        "collection_name": "US_LAWS",
        "collection_distance_metric": "Cosine",
    }
}


class Config:
    model_db_config = {}

    def use_config(self, mode: Literal["online", "offline"] = "online", llm: str = None):
        try:
            config_module = config_mode["online"]
            # For debugging purposes, remove in production
            self.model_db_config = config_module
        except ModuleNotFoundError:
            logging.error(f"Could not find config named {mode}, going ahead with defaults")

        self._init_run_config(self.model_db_config)

    def _init_run_config(self, model_db_config):
        self.collection_name = model_db_config.get("collection_name", "US_LAWS")
        self.collection_distance_metric = model_db_config.get(
            "collection_distance_metric", "Cosine"
        )
        self.llm = model_db_config.get("llm", "None")
        self.embeddings = model_db_config.get("embeddings", "None")
        self.db_url = model_db_config.get("db_url", "localhost")
        self.db_port = model_db_config.get("db_port", 6333)
