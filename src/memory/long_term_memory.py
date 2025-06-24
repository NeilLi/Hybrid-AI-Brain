#!/usr/bin/env python3
"""
src/memory/long_term_memory.py

Defines the Long-Term Memory (L) component, using ChromaDB and sentence-transformers.
"""

from typing import List, Any, Optional, Dict
import logging

class LongTermMemory:
    """
    Long-Term Memory with semantic search using ChromaDB and sentence-transformers.
    """
    def __init__(self, persist_directory: Optional[str] = None, embedding_model_name: str = "all-MiniLM-L6-v2"):
        """
        Args:
            persist_directory: Where to store ChromaDB (if None, runs in-memory).
            embedding_model_name: Name of the sentence-transformers model to use.
        """
        self._is_backend_ready = False
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            from chromadb.config import Settings

            self.chroma_client = chromadb.Client(Settings(
                persist_directory=persist_directory or ".chromadb_ltm"
            ))
            self.collection = self.chroma_client.get_or_create_collection("long_term_memory")
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self._is_backend_ready = True
            logging.info("LongTermMemory: Using ChromaDB backend for storage.")
        except ImportError:
            self.chroma_client = None
            self.collection = None
            self.embedding_model = None
            self._store: List[Any] = []
            logging.warning("LongTermMemory: ChromaDB/sentence-transformers not installed. Using in-memory fallback.")

    def add_knowledge(self, item: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a knowledge item (string) to long-term memory.
        Args:
            item: The content (text) to store.
            metadata: Optional dictionary of metadata for the item.
        """
        if self._is_backend_ready:
            emb = self.embedding_model.encode(item)
            _id = str(hash(item))  # Use hash for deterministic, unique ID
            self.collection.upsert(
                ids=[_id],
                embeddings=[emb.tolist()],
                documents=[item],
                metadatas=[metadata or {}]
            )
            logging.info("LongTermMemory: Added item to ChromaDB.")
        else:
            self._store.append(item)
            logging.info("LongTermMemory: Added item to in-memory store.")

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k semantically relevant items for a given query.
        Args:
            query: Query string.
            top_k: Maximum number of items to return.
        Returns:
            List of matching documents.
        """
        if self._is_backend_ready:
            emb = self.embedding_model.encode(query)
            results = self.collection.query(
                query_embeddings=[emb.tolist()],
                n_results=top_k
            )
            # Results: dict with keys: 'ids', 'documents', 'distances', 'metadatas'
            docs = results.get("documents", [[]])[0]
            logging.info(f"LongTermMemory: Found {len(docs)} relevant item(s) with ChromaDB.")
            return docs
        else:
            # Simple text match fallback
            relevant = [item for item in self._store if query.lower() in item.lower()]
            logging.info(f"LongTermMemory: Found {len(relevant)} relevant item(s) in fallback mode.")
            return relevant[:top_k]

    @property
    def total_items(self) -> int:
        if self._is_backend_ready:
            # Use ChromaDB collection count (estimate)
            try:
                return len(self.collection.get()["ids"])
            except Exception:
                return 0
        else:
            return len(self._store)

    def __repr__(self):
        return f"LongTermMemory(total_items={self.total_items}, backend={'ChromaDB' if self._is_backend_ready else 'in-memory'})"

# --- Demo Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ltm = LongTermMemory()
    ltm.add_knowledge("Paris is the capital of France.")
    ltm.add_knowledge("Python is a popular programming language.")
    ltm.add_knowledge("The FIFA World Cup was last won by Argentina.")
    print("Items stored:", ltm.total_items)
    print("Query: 'programming'")
    print("Relevant:", ltm.retrieve_relevant_knowledge("programming"))
    print("Query: 'World Cup'")
    print("Relevant:", ltm.retrieve_relevant_knowledge("World Cup"))
