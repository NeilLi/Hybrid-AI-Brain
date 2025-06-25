#!/usr/bin/env python3
"""
src/memory/long_term_memory.py

Defines the Long-Term Memory (L) component, using ChromaDB with persistent local storage
and sentence-transformers for semantic search capabilities.
Fixed to handle ChromaDB singleton issues and multiple instances properly.
"""

from typing import List, Any, Optional, Dict, Tuple
import logging
import uuid
import time
import os
from pathlib import Path

# Global ChromaDB client cache to avoid singleton conflicts
_chroma_clients = {}

class LongTermMemory:
    """
    Long-Term Memory with semantic search using ChromaDB with persistent local storage.
    """

    def __init__(
        self,
        persist_directory: Optional[str] = None,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "long_term_memory"
    ):
        self._is_backend_ready = False
        self._store = []
        
        if persist_directory is None:
            persist_directory = os.path.join(".", "data", "chroma_db")
        
        self.persist_directory = Path(persist_directory)
        self.persist_directory.mkdir(parents=True, exist_ok=True)
        self.collection_name = self._sanitize_collection_name(collection_name)
        
        try:
            self._initialize_chromadb(embedding_model_name)
        except Exception as e:
            logging.warning(f"LongTermMemory: ChromaDB/sentence-transformers not installed or failed to init ({e}). Falling back to in-memory list.")
            self._fallback_to_memory()

    def _initialize_chromadb(self, embedding_model_name: str):
        """Initialize ChromaDB backend with proper singleton handling."""
        import chromadb
        from sentence_transformers import SentenceTransformer
        
        client_key = str(self.persist_directory.absolute())
        
        if client_key in _chroma_clients:
            self.chroma_client = _chroma_clients[client_key]
        else:
            self.chroma_client = chromadb.PersistentClient(path=str(self.persist_directory))
            _chroma_clients[client_key] = self.chroma_client
        
        self.collection = self.chroma_client.get_or_create_collection(self.collection_name)
        logging.info(f"LongTermMemory: Loaded/Created collection '{self.collection_name}' with {self.collection.count()} items.")
        
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self._is_backend_ready = True
        logging.info(f"LongTermMemory: Using ChromaDB persistent backend at '{self.persist_directory.absolute()}'")

    def _fallback_to_memory(self):
        """Set up in-memory fallback storage."""
        self.collection = None
        self.chroma_client = None
        self.embedding_model = None
        self._is_backend_ready = False

    def _sanitize_collection_name(self, name: str) -> str:
        """Sanitize collection name to ensure it's valid for ChromaDB."""
        import re
        return re.sub(r'[^a-zA-Z0-9_-]', '_', name)

    def add_knowledge(self, item: str, metadata: Optional[Dict[str, Any]] = None):
        """Add a knowledge item (string) to long-term memory."""
        if not item or not item.strip():
            return
            
        if self._is_backend_ready and self.collection:
            unique_id = f"item_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
            meta = (metadata or {}).copy()
            meta.setdefault("timestamp", time.time())
            meta.setdefault("content_length", len(item))
            
            self.collection.upsert(documents=[item], metadatas=[meta], ids=[unique_id])
        else:
            self._store.append(item)

    def retrieve_relevant_knowledge(self, query: str, top_k: int = 3) -> List[str]:
        """Retrieves top-k semantically relevant items for a given query."""
        if not query or not query.strip() or self.total_items == 0:
            return []
            
        if self._is_backend_ready and self.collection:
            emb = self.embedding_model.encode(query)
            results = self.collection.query(query_embeddings=[emb.tolist()], n_results=min(top_k, self.total_items), include=["documents"])
            return results.get("documents", [[]])[0]
        else:
            return self._simple_text_search(query, top_k)

    def _simple_text_search(self, query: str, top_k: int) -> List[str]:
        """Fallback simple text search for in-memory store."""
        query_lower = query.lower()
        return [item for item in self._store if query_lower in item.lower()][:top_k]

    @property
    def total_items(self) -> int:
        """Get the total number of items in storage."""
        if self._is_backend_ready and self.collection:
            return self.collection.count()
        return len(self._store)

    def clear(self):
        """Erase every vector in this collection."""
        if self._is_backend_ready and hasattr(self, 'chroma_client'):
            try:
                self.chroma_client.delete_collection(self.collection_name)
            except Exception:
                pass 
            self.collection = self.chroma_client.create_collection(self.collection_name)
        else:
            self._store.clear()

    def get_all_items(self) -> List[str]:
        """Return all stored documents (no metadata)."""
        if not self._is_backend_ready or self.total_items == 0:
            return []
        data = self.collection.get(include=["documents"])
        return data.get("documents", [])

    def get_metadata_for_query(self, query: str, top_k: int = 1) -> List[Dict[str, Any]]:
        """
        Return a list of dictionaries, each containing the document and its metadata.
        """
        if not self._is_backend_ready or not query.strip() or self.total_items == 0:
            return []
        
        # --- FIX: Query for both documents and metadatas ---
        res = self.collection.query(
            query_texts=[query],
            n_results=min(top_k, self.total_items),
            include=["documents", "metadatas"],
        )
        
        # --- FIX: Zip the results into the expected structure ---
        docs  = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        return [{"document": d, "metadata": m} for d, m in zip(docs, metas)]

    @classmethod
    def reset_global_state(cls):
        """Reset global ChromaDB client cache. Useful for testing."""
        global _chroma_clients
        _chroma_clients.clear()
