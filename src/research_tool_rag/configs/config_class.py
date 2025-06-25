import importlib
import logging
from typing import Literal

# from src.research_tool_rag.configs.defaults import defaults


class Config:
    model_db_config = {}

    def use_config(self, mode: Literal["online", "offline"] = "online", llm: str = None):
        try:
            config_module = importlib.import_module(f"{__package__}.{mode}")
            self.model_db_config = config_module.model_db_config
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
