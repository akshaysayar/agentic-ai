from research_tool_rag.rag.ingest_data import Hierarchy
from lxml import etree
from pathlib import Path
from research_tool_rag.utils.utils import setup_logging
from research_tool_rag.rag.qdrant import QdrantDB
import logging
import argparse

logger = logging.getLogger(__name__)


setup_logging("ingest_data" , stream_handler=True)

logger.info("Starting to build hierarchy from XML file.")



# h = Hierarchy(path="/home/akshaysayar/agentic_ai/data/00.raw/ny-laws/abandoned_property/fixtures/02.purged/2024/0420-000000/ny/statute/xml/abandoned_property.xml")
# h.build_hierarchy()
qdrant_db = QdrantDB()
# breakpoint()
# qdrant_db.index_sections(h.children)
oh_yeah = qdrant_db.search(query="Unclaimed amounts or securities held by foreign corporations")
breakpoint()

def main(content_set: str, online_model: bool):

    if online_model:
        logger.info(f"Starting ingestion for content set: {content_set} with online model.")
        qdrant_db = QdrantDB("US_Laws_online")
        
    else:
        logger.info(f"Starting ingestion for content set: {content_set} with offline model.")
        qdrant_db = QdrantDB("US_Laws_offline")
        
    
    
    for files in Path(__file__).parent.parent.glob(f"data/00.raw/{content_set}/**/*.xml"):
        logger.info(f"Processing file: {files}")
        title = Hierarchy(path=files)
        title.build_hierarchy()
        qdrant_db.index_sections(title.children)
        logger.info(f"Indexed sections from {files} into Qdrant.")

    qdrant_db.index_sections(h.children)
    logger.info("Hierarchy built and indexed in Qdrant.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest data from XML files into Qdrant.")
    parser.add_argument("--content-set", type=str, required=True, help="Which content set to ingest.")
    parser.add_argument("--online_model", type=bool, default=False, help="Use online model for processing.")
    args = parser.parse_args()
    main(args.xml_path, args.online_model)