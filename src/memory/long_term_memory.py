#!/usr/bin/env python3
"""
src/memory/long_term_memory.py

Defines the Long-Term Memory (L) component, using ChromaDB and sentence-transformers.
Fixed to handle test isolation and proper ID generation.
"""

from typing import List, Any, Optional, Dict
import logging
import uuid
import time

class LongTermMemory:
    """
    Long-Term Memory with semantic search using ChromaDB and sentence-transformers.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: Optional[str] = None,
    ):
        """
        Args:
            persist_directory: Where to store ChromaDB (if None, runs in-memory).
            embedding_model_name: Name of the sentence-transformers model to use.
            collection_name: Name of the collection. If None, generates a unique name.
        """
        self._is_backend_ready = False
        self._store = []
        
        # Generate unique collection name to avoid conflicts between tests
        if collection_name is None:
            collection_name = f"long_term_memory_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        try:
            import chromadb
            from sentence_transformers import SentenceTransformer
            from chromadb.config import Settings

            # Use in-memory mode for tests unless explicitly specified
            if persist_directory:
                settings = Settings(persist_directory=persist_directory)
            else:
                settings = Settings()  # in-memory mode

            self.chroma_client = chromadb.Client(settings)
            self.collection_name = collection_name
            self.collection = self.chroma_client.get_or_create_collection(collection_name)
            self.embedding_model = SentenceTransformer(embedding_model_name)
            self._is_backend_ready = True
            logging.info(f"LongTermMemory: Using ChromaDB backend with collection '{collection_name}'.")
        except ImportError:
            self.chroma_client = None
            self.collection = None
            self.embedding_model = None
            self.collection_name = None
            self._store: List[Any] = []
            logging.warning(
                "LongTermMemory: ChromaDB/sentence-transformers not installed. Using in-memory fallback."
            )

    def add_knowledge(self, item: str, metadata: Optional[Dict[str, Any]] = None):
        """
        Add a knowledge item (string) to long-term memory.
        Args:
            item: The content (text) to store.
            metadata: Optional dictionary of metadata for the item.
        """
        if not item or not item.strip():
            logging.warning("LongTermMemory: Ignoring empty or whitespace-only item.")
            return
            
        if self._is_backend_ready:
            try:
                # Generate unique ID to avoid conflicts
                unique_id = f"item_{uuid.uuid4().hex}"
                
                # Prepare metadata
                if metadata is None:
                    metadata = {}
                metadata.update({
                    "timestamp": time.time(),
                    "content_hash": str(hash(item))
                })
                
                # Add to ChromaDB
                self.collection.upsert(
                    documents=[item],
                    metadatas=[metadata],
                    ids=[unique_id],
                )
                logging.debug(f"LongTermMemory: Added item to ChromaDB with ID {unique_id}.")
            except Exception as e:
                logging.error(f"LongTermMemory: Failed to add item to ChromaDB: {e}")
                # Fallback to in-memory store
                self._store.append(item)
        else:
            self._store.append(item)
            logging.debug("LongTermMemory: Added item to in-memory store.")

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """
        Retrieves top-k semantically relevant items for a given query.
        Args:
            query: Query string.
            top_k: Maximum number of items to return.
        Returns:
            List of matching documents.
        """
        if not query or not query.strip():
            return []
            
        if self._is_backend_ready:
            try:
                # Check if collection has any items
                if self.total_items == 0:
                    return []
                    
                emb = self.embedding_model.encode(query)
                results = self.collection.query(
                    query_embeddings=[emb.tolist()], 
                    n_results=min(top_k, self.total_items)
                )
                
                # Extract documents from results
                docs = results.get("documents", [[]])[0] if results.get("documents") else []
                logging.debug(f"LongTermMemory: Found {len(docs)} relevant item(s) with ChromaDB.")
                return docs
            except Exception as e:
                logging.error(f"LongTermMemory: Error during ChromaDB query: {e}")
                # Fallback to simple search
                return self._simple_text_search(query, top_k)
        else:
            return self._simple_text_search(query, top_k)

    def _simple_text_search(self, query: str, top_k: int) -> List[str]:
        """Fallback simple text search for in-memory store."""
        query_lower = query.lower()
        relevant = [item for item in self._store if query_lower in item.lower()]
        logging.debug(f"LongTermMemory: Found {len(relevant)} relevant item(s) in fallback mode.")
        return relevant[:top_k]

    @property
    def total_items(self) -> int:
        if self._is_backend_ready:
            try:
                count = self.collection.count()
                return count if count is not None else 0
            except Exception as e:
                logging.error(f"LongTermMemory: Error getting item count: {e}")
                return 0
        else:
            return len(self._store)

    def clear(self):
        """Clear all items from memory."""
        if self._is_backend_ready:
            try:
                # Delete the collection and recreate it
                self.chroma_client.delete_collection(self.collection_name)
                self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
                logging.info("LongTermMemory: Cleared ChromaDB collection.")
            except Exception as e:
                logging.error(f"LongTermMemory: Error clearing ChromaDB collection: {e}")
        else:
            self._store.clear()
            logging.info("LongTermMemory: Cleared in-memory store.")

    def get_all_items(self) -> List[str]:
        """Get all items in memory (for testing/debugging)."""
        if self._is_backend_ready:
            try:
                if self.total_items == 0:
                    return []
                results = self.collection.get()
                return results.get("documents", [])
            except Exception as e:
                logging.error(f"LongTermMemory: Error getting all items: {e}")
                return []
        else:
            return self._store.copy()

    def __repr__(self):
        backend = "ChromaDB" if self._is_backend_ready else "in-memory"
        return f"LongTermMemory(total_items={self.total_items}, backend={backend})"


# --- Demo Block ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ltm = LongTermMemory()
    
    print(f"Initial state: {ltm}")
    
    ltm.add_knowledge("Paris is the capital of France.")
    ltm.add_knowledge("Python is a popular programming language.")
    ltm.add_knowledge("The FIFA World Cup was last won by Argentina.")
    
    print(f"After adding items: {ltm}")
    print("Items stored:", ltm.total_items)
    
    print("\nQuery: 'programming'")
    results = ltm.retrieve_relevant_knowledge("programming")
    print("Relevant:", results)
    
    print("\nQuery: 'World Cup'")
    results = ltm.retrieve_relevant_knowledge("World Cup")
    print("Relevant:", results)
    
    print("\nAll items:", ltm.get_all_items())