"""
Vector store operations using ChromaDB.
Handles storage, retrieval, and management of document embeddings.
"""

import os
from typing import Optional

import chromadb
import numpy as np

from chunker import Chunk


class VectorStore:
    """ChromaDB-backed vector store for document chunks."""

    def __init__(
        self,
        persist_directory: str = "data/vectorstore",
        collection_name: str = "documents",
    ):
        """
        Initialize ChromaDB vector store.
        
        Args:
            persist_directory: Where to save the database on disk
            collection_name: Name of the collection
        """
        os.makedirs(persist_directory, exist_ok=True)

        self.client = chromadb.PersistentClient(path=persist_directory)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"Vector store initialized: {self.collection.count()} existing chunks")

    def add_chunks(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
    ):
        """
        Add document chunks with their embeddings to the store.
        
        Args:
            chunks: List of Chunk objects
            embeddings: numpy array of shape (len(chunks), dim)
        """
        ids = [chunk.chunk_id for chunk in chunks]
        documents = [chunk.text for chunk in chunks]
        metadatas = [
            {
                "document_name": chunk.document_name,
                "page_number": chunk.page_number,
                "start_char": chunk.start_char,
                "end_char": chunk.end_char,
                "source_label": chunk.source_label,
            }
            for chunk in chunks
        ]

        # ChromaDB has a batch limit, so add in batches
        batch_size = 500
        for i in range(0, len(ids), batch_size):
            end = min(i + batch_size, len(ids))
            self.collection.add(
                ids=ids[i:end],
                documents=documents[i:end],
                metadatas=metadatas[i:end],
                embeddings=embeddings[i:end].tolist(),
            )

        print(f"Added {len(chunks)} chunks. Total: {self.collection.count()}")

    def query(
        self,
        query_embedding: np.ndarray,
        top_k: int = 10,
        filter_doc: Optional[str] = None,
    ) -> list[dict]:
        """
        Query the vector store for similar chunks.
        
        Args:
            query_embedding: Query vector
            top_k: Number of results to return
            filter_doc: Optional document name to filter by
        
        Returns:
            List of dicts with 'text', 'metadata', 'distance'
        """
        where = {"document_name": filter_doc} if filter_doc else None

        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where=where,
            include=["documents", "metadatas", "distances"],
        )

        # Flatten results into list of dicts
        output = []
        for i in range(len(results["ids"][0])):
            output.append({
                "id": results["ids"][0][i],
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
                "score": 1 - results["distances"][0][i],  # Convert distance to similarity
            })

        return output

    def delete_document(self, document_name: str):
        """Remove all chunks from a specific document."""
        self.collection.delete(
            where={"document_name": document_name}
        )
        print(f"Deleted chunks for '{document_name}'. Remaining: {self.collection.count()}")

    def clear(self):
        """Remove all chunks from the store."""
        self.client.delete_collection(self.collection.name)
        self.collection = self.client.get_or_create_collection(
            name=self.collection.name,
            metadata={"hnsw:space": "cosine"},
        )
        print("Vector store cleared")

    def get_document_names(self) -> list[str]:
        """Get list of unique document names in the store."""
        results = self.collection.get(include=["metadatas"])
        if not results["metadatas"]:
            return []
        names = set(m["document_name"] for m in results["metadatas"])
        return sorted(names)

    def get_all(self):
        """Get all documents, metadatas, and ids from the store."""
        results = self.collection.get(include=["documents", "metadatas"])
        return results
