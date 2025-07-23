import argparse
import logging
from pathlib import Path
import time

from langchain_core.documents import Document

from research_tool_rag.configs import config
from research_tool_rag.db_store.qdrant import QdrantDB
from research_tool_rag.preprocessing.hierarchy import Hierarchy
from research_tool_rag.utils.utils import setup_logging

logger = logging.getLogger(__name__)


setup_logging("ingest_data", stream_handler=True)

logger.info("Starting to build hierarchy from XML file.")


# h = Hierarchy(path="/home/akshaysayar/agentic_ai/data/00.raw/ny-laws/abandoned_property/fixtures/02.purged/2024/0420-000000/ny/statute/xml/abandoned_property.xml")
# h.build_hierarchy()
# qdrant_db = QdrantDB()
# breakpoint()
# qdrant_db.index_sections(h.children)
# oh_yeah = qdrant_db.search(query="Unclaimed amounts or securities held by foreign corporations")
# breakpoint()


# def init_model():
#     import getpass
#     import os

#     if not os.environ.get("GOOGLE_API_KEY"):
#         os.environ["GOOGLE_API_KEY"] = "AIzaSyDB5-l0ck6YwuCyLxDHYl76Jt4pOpDB5Zw"#getpass.getpass("Enter API key for Google Gemini: ")

#     from langchain.chat_models import init_chat_model

#     llm = init_chat_model("gemini-2.0-flash", model_provider="google_genai")

#     from langchain_google_genai import GoogleGenerativeAIEmbeddings

#     embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
#     return llm, embeddings

# def init_db(embeddings):
#     from langchain_qdrant import QdrantVectorStore
#     from qdrant_client import QdrantClient

#     client = QdrantClient("localhost", port=6333)
#     collections = [c.name for c in client.get_collections().collections]
#     collection_name = "US_LAWS"

#     if collection_name not in collections:
#             client.recreate_collection(
#                 collection_name=collection_name,
#                 vectors_config=models.VectorParams(size=768, distance="Cosine"),
#             )

#     vector_store = QdrantVectorStore(
#         client=client,
#         collection_name="US_LAWS",
#         embedding=embeddings,
#     )
#     return client, vector_store

# def main(content_set: str, online_model: bool = False):
#     llm, embeddings = init_model()
#     db, vector_store = init_db(embeddings)
#     # path = Path(__file__).parent.glob(f"data/00.raw/{content_set}/abandoned_property/**/*.xml")
#     # breakpoint()

#     for files in Path(__file__).parent.glob(f"data/00.raw/{content_set}/*/**/*.xml"):
#         logger.info(f"Processing file: {files}")
#         title = Hierarchy(path=files)
#         title.build_hierarchy()
#         docs = []
#         for section in title.children:
#              idx= 0
#              docs.append(Document(
#                 page_content=section.content,
#                 metadata={
#                         "section_id": str(section.id),
#                         "number": section.number,
#                         "name": section.name,
#                         "state": section.hierarchy.state,
#                         "law_type": section.hierarchy.law_type,
#                         "title": section.hierarchical_title,
#                         "hierarchical_name": section.hierarchical_name,
#                         "hierarchical_number": section.hierarchical_number,
#                         "paragraph_id": f"{str(section.id)}_{idx}",
#                 }
#             ))

#         logger.info(f"Indexed sections from {files} into Qdrant.")

#     # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
#     # all_splits = text_splitter.split_documents(docs)

#     # Index chunks
#         _ = vector_store.add_documents(documents=docs)


def process_and_ingest(content_set):
    l = ['judiciary.xml','legislative.xml','defense_emergency_act_1951.xml','regulation_of_lobbying_act.xml','canal.xml','retirement_and_social_security.xml','state_administrative_procedure_act.xml',
         'emergency_housing_rent_control_law.xml','real_property_actions_and_proceedings.xml','nys_project_finance_agency_act.xml','second_class_cities.xml','personal_property.xml',
         'surrogates_court_procedure.xml','civil_practice_law_and_rules.xml','lien.xml']
    not_done = []
    qdb = QdrantDB()
    for files in Path(__file__).parent.parent.parent.parent.glob(
        f"data/00.raw/{content_set}/*/**/*.xml"
    ):
        if files.name in l:
            continue
        logger.info(f"Processing file: {files}")
        title = Hierarchy(path=files)
        title.build_hierarchy()
        docs = []
        for section in title.children:
            idx = 0
            docs.append(
                Document(
                    page_content=section.content,
                    metadata={
                        "section_id": str(section.id),
                        "number": section.number,
                        "name": section.name,
                        "state": section.hierarchy.state,
                        "law_type": section.hierarchy.law_type,
                        "title": section.hierarchical_title,
                        "hierarchical_name": section.hierarchical_name,
                        "hierarchical_number": section.hierarchical_number,
                        "paragraph_id": f"{str(section.id)}_{idx}",
                    },
                )
            )

        logger.info(f"Indexed sections from {files} into Qdrant.")

        # text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        # all_splits = text_splitter.split_documents(docs)

        # Index chunks
        try:
            for doc in docs:
                _ = qdb.vector_store.add_documents(documents=[doc])
                time.sleep(0.1)
            l.append(files.name)
                
        except Exception as e:
            print(f"Error indexing {files}: {e}")
            print(l)
            raise e
                # not_done = not_done.append(files.name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data from XML files into Qdrant.")
    parser.add_argument(
        "--content_set", type=str, required=True, help="Which content set to ingest."
    )
    # parser.add_argument("--online_model", type=bool, default=False, help="Use online model for processing.")
    args = parser.parse_args()
    config.use_config("online")
    process_and_ingest(args.content_set)
