import logging
from collections import namedtuple

from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models

from research_tool_rag.configs import config

logger = logging.getLogger(__name__)

SearchResult = namedtuple("SearchResult", ["name", "content"])


class QdrantDB:
    client: QdrantClient
    collection_name: str
    vector_store: QdrantVectorStore

    def __init__(self, collection_name=None):
        # Use provided collection_name, else config, else default
        try:
            # from research_tool_rag.configs import config
            config_collection = getattr(config, "collection_name", None)
        except Exception:
            config_collection = None
        self.collection_name = collection_name or config_collection or "PROFILE_COLLECTION"
        # Initialize Qdrant with defaults if config is missing values
        db_url = getattr(config, "db_url", "localhost")
        db_port = getattr(config, "db_port", 6333)
        self.client = QdrantClient(db_url, port=db_port)

        # Always ensure collection has correct vector size
        expected_size = 768
        # Always force recreate the collection with the correct vector size before using it
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=expected_size,
                distance=getattr(config, "model_db_config", {}).get(
                    "collection_distance", "Cosine"
                ),
            ),
        )

        # Try to get embeddings from config.model_db_config if available
        embeddings = None
        if hasattr(config, "model_db_config") and isinstance(config.model_db_config, dict):
            embeddings = config.model_db_config.get("embeddings", None)
        if embeddings is None:
            embeddings = getattr(config, "embeddings", None)
        if embeddings is None:
            raise ValueError(
                "No embedding model found in config or model_db_config. Please set one."
            )
        self.vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=embeddings,
        )

    # @staticmethod
    # def chunk_section(texts: List[str], max_tokens=512, overlap=50):
    #     """Split section into chunks while retaining IDs."""

    #     chunks = []
    #     for text in texts:
    #         # # Tokenize text without truncation to get all tokens
    #         # inputs = llm_tokenizer(text, return_tensors='pt', padding=False, truncation=False)
    #         # input_ids = inputs['input_ids'][0]  # tensor of token ids for this text

    #         # start = 0
    #         # while start < len(input_ids):
    #         #     end = start + max_tokens
    #         #     chunk_ids = input_ids[start:end]
    #         #     chunk_text = llm_tokenizer.decode(chunk_ids, skip_special_tokens=True)
    #         #     yield (chunk_ids, chunk_text.strip())
    #         #     start += max_tokens - overlap

    #         tokens = tokenizer(text, return_tensors="pt")
    #         input_ids = tokens["input_ids"].squeeze(0)  # [batch, seq] -> [seq]
    #         attention_masks = tokens["attention_mask"].squeeze(0)

    #         for i in range(0, len(input_ids) - max_tokens + 1, max_tokens - overlap):
    #             chunk_ids = input_ids[i : i + max_tokens]
    #             chunk_masks = attention_masks[i : i + max_tokens]
    #             chunks.append((chunk_ids, chunk_masks))

    #         # Handle last chunk if it's shorter
    #         if len(input_ids) % (max_tokens - overlap) != 0:
    #             chunk_ids = input_ids[-max_tokens:]
    #             chunk_masks = attention_masks[-max_tokens:]
    #             chunks.append((chunk_ids, chunk_masks))

    #     for chunk in chunks:
    #         yield chunk

    # def embed(self, text) -> List[float]:
    #     """Create embetextdding for a piece of text."""
    #     inputs = tokenizer(text, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     # Average pool
    #     return outputs.last_hidden_state.mean(dim=1)

    # def index_sections(self, sections: List[Node]):
    #     """Index sections into Qdrant with metadata linking back to their IDs."""
    #     sections = sections
    #     points = []
    #     for section in sections:
    #         section_id = str(section.id)
    #         logger.info(f"Indexing section {section.hierarchical_name}")
    #         for idx, chunk in enumerate(
    #             self.chunk_section([section.hierarchical_name, section.content])
    #         ):
    #             chunk_ids, chunk_masks = chunk
    #             # vector = self.embed(chunk)
    #             with torch.no_grad():
    #                 outputs = model(
    #                     input_ids=chunk_ids.unsqueeze(0), attention_mask=chunk_masks.unsqueeze(0)
    #                 )
    #                 emb = outputs.last_hidden_state.mean(dim=1)
    #             text = tokenizer.decode(chunk_ids, skip_special_tokens=True)
    #             if idx == 0:
    #                 text = f"{text} ----- "
    #             vector = emb
    #             point = PointStruct(
    #                 id=str(uuid.uuid4()),
    #                 # Convert tensor to list
    #                 vector=vector[0],
    #                 payload={
    #                     "section_id": section_id,
    #                     "number": section.number,
    #                     "name": section.name,
    #                     "state": section.hierarchy.state,
    #                     "law_type": section.hierarchy.law_type,
    #                     "title": section.hierarchical_title,
    #                     "hierarchical_name": section.hierarchical_name,
    #                     "hierarchical_number": section.hierarchical_number,
    #                     "paragraph_id": f"{section_id}_{idx}",
    #                     "paragraph_content": text,
    #                 },
    #             )
    #             # Batch-upsert
    #             self.client.upsert(collection_name=self.collection_name, points=[point])

    # def search(self, query, collection_name="US_Laws", top=5):
    #     """Search Qdrant for semantic match to a query."""
    #     result = []
    #     inputs = tokenizer(query, return_tensors="pt")
    #     with torch.no_grad():
    #         outputs = model(**inputs)
    #     embedding = outputs.last_hidden_state.mean(dim=1)

    #     semantic_search_res = self.client.search(
    #         collection_name=collection_name, query_vector=embedding[0], limit=top, with_payload=True
    #     )

    #     semantic_search_res = list(semantic_search_res)
    #     semantic_search_res.sort(key=lambda x: x.score, reverse=True)
    #     result_section_ids = set([res.payload["section_id"] for res in semantic_search_res])

    #     for section_id in result_section_ids:

    #         filter_condition = Filter(
    #             must=[FieldCondition(key="section_id", match=MatchValue(value=section_id))]
    #         )

    #         # Perform semantic search with filter
    #         res = self.client.scroll(
    #             collection_name=self.collection_name,
    #             limit=100,
    #             scroll_filter=filter_condition,
    #         )[0]
    #         res.sort(key=lambda x: x.payload["paragraph_id"])

    #         name = res[0].payload["hierarchical_name"]
    #         content = (" ".join([r.payload["paragraph_content"] for r in res])).split(" ----- ")[1]
    #         result.append(SearchResult(name, content))

    #     return result

    # # example usage:
    # results = search("abandoned property procedures")
    # for res in results:
    #     print("Chunk:", res.payload['paragraph_content'])
    #     print("Section IDs!", res.payload['section_id'])
