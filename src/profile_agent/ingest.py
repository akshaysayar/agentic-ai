import logging

from langchain_core.documents import Document

from research_tool_rag.db_store.qdrant import QdrantDB

logger = logging.getLogger(__name__)


def ingest_profile(profile_path: str, collection_name: str = None):
    # Try to get collection_name from config, else use default or passed value
    try:
        from research_tool_rag.configs import config

        config_collection = getattr(config, "collection_name", None)
    except Exception:
        config_collection = None
    collection = collection_name or config_collection or "PROFILE_COLLECTION"
    qdb = QdrantDB()
    if hasattr(qdb, "collection_name"):
        qdb.collection_name = collection
    with open(profile_path, "r") as f:
        json_str = f.read()
    doc = Document(page_content=json_str, metadata={"source": str(profile_path)})
    logger.info(
        f"Indexing entire profile JSON from {profile_path} into Qdrant collection '{collection}'."
    )
    qdb.vector_store.add_documents(documents=[doc])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest Akshay_Sayar_CV.json into Qdrant.")
    parser.add_argument(
        "--profile", type=str, default="data/Akshay_Sayar_CV.json", help="Path to profile JSON"
    )
    parser.add_argument(
        "--collection", type=str, default=None, help="Qdrant collection name (optional)"
    )
    args = parser.parse_args()
    # Explicitly use the online config for embeddings and db settings
    from research_tool_rag.configs import config

    config.use_config("online")
    ingest_profile(args.profile, collection_name=args.collection)
