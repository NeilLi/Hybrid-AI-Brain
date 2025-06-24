import chromadb
from chromadb.config import Settings

client = chromadb.Client(Settings())  # ← No persist_directory! In-memory by default.
collection = client.get_or_create_collection("my_collection")
collection.upsert(
    documents=["Hello world"],
    metadatas=[{"foo": "bar"}],
    ids=["doc1"],
)
print(collection.get())
