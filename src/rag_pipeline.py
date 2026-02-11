"""
End-to-end RAG pipeline orchestrating ingestion and querying.
"""

import yaml

from document_loader import Document, load_pdf, load_uploaded_pdf
from chunker import chunk_document
from embeddings import EmbeddingModel
from vector_store import VectorStore
from retriever import HybridRetriever
from generator import AnswerGenerator


def load_config(config_path: str = "configs/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


class RAGPipeline:
    """
    Complete RAG pipeline: ingest documents, answer questions with citations.
    
    Usage:
        pipeline = RAGPipeline()
        pipeline.ingest_pdf("report.pdf")
        result = pipeline.query("What were Q3 revenues?")
        print(result["answer"])
        print(result["sources"])
    """

    def __init__(self, config_path: str = "configs/config.yaml"):
        self.config = load_config(config_path)

        # Initialize components
        self.embedding_model = EmbeddingModel(
            model_name=self.config["embeddings"]["model_name"],
            device=self.config["embeddings"]["device"],
            normalize=self.config["embeddings"]["normalize"],
        )

        self.vector_store = VectorStore(
            persist_directory=self.config["vector_store"]["persist_directory"],
            collection_name=self.config["vector_store"]["collection_name"],
        )

        self.retriever = HybridRetriever(
            embedding_model=self.embedding_model,
            vector_store=self.vector_store,
            use_hybrid=self.config["retrieval"]["use_hybrid"],
            semantic_weight=self.config["retrieval"]["semantic_weight"],
            bm25_weight=self.config["retrieval"]["bm25_weight"],
            reranker_model=self.config["retrieval"]["reranker_model"],
        )

        self.generator = AnswerGenerator(
            provider=self.config["generation"]["provider"],
            model=self.config["generation"]["model"],
            temperature=self.config["generation"]["temperature"],
            max_tokens=self.config["generation"]["max_tokens"],
            config_path=config_path,
        )

        self.conversation_history = []
        print("RAG Pipeline initialized!")

    def ingest_pdf(self, file_path: str) -> dict:
        """
        Ingest a PDF file into the pipeline.
        
        Steps: Parse → Chunk → Embed → Store
        
        Returns:
            Dict with ingestion stats
        """
        # Parse
        document = load_pdf(file_path)
        print(f"Parsed '{document.name}': {document.total_pages} pages")

        return self._ingest_document(document)

    def ingest_uploaded_pdf(self, file_bytes: bytes, filename: str) -> dict:
        """Ingest a PDF from uploaded bytes (for Streamlit)."""
        document = load_uploaded_pdf(file_bytes, filename)
        return self._ingest_document(document)

    def _ingest_document(self, document: Document) -> dict:
        """Internal: chunk, embed, and store a document."""
        # Chunk
        chunks = chunk_document(
            document,
            strategy=self.config["chunking"]["strategy"],
            chunk_size=self.config["chunking"]["chunk_size"],
            chunk_overlap=self.config["chunking"]["chunk_overlap"],
        )

        # Embed
        embeddings = self.embedding_model.embed_chunks(
            chunks, batch_size=self.config["embeddings"]["batch_size"]
        )

        # Store
        self.vector_store.add_chunks(chunks, embeddings)

        # Refresh BM25 index
        self.retriever.refresh_bm25_index()

        return {
            "document_name": document.name,
            "total_pages": document.total_pages,
            "num_chunks": len(chunks),
            "avg_chunk_size": sum(len(c.text) for c in chunks) / len(chunks) if chunks else 0,
        }

    def query(self, question: str, filter_doc: str = None) -> dict:
        """
        Ask a question and get a cited answer.
        
        Args:
            question: Natural language question
            filter_doc: Optional document name to search within
        
        Returns:
            Dict with 'answer', 'sources', 'source_map'
        """
        # Retrieve
        retrieved = self.retriever.retrieve(
            query=question,
            top_k=self.config["retrieval"]["top_k"],
            rerank_top_k=self.config["retrieval"]["rerank_top_k"],
            filter_doc=filter_doc,
        )

        # Generate
        result = self.generator.generate(
            query=question,
            retrieved_chunks=retrieved,
            conversation_history=self.conversation_history,
            system_prompt=self.config["generation"]["system_prompt"],
        )

        # Update conversation history
        self.conversation_history.append({
            "question": question,
            "answer": result["answer"],
        })

        # Trim history
        max_history = self.config["conversation"]["max_history"]
        if len(self.conversation_history) > max_history:
            self.conversation_history = self.conversation_history[-max_history:]

        return result

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []

    def get_documents(self) -> list[str]:
        """Get list of ingested document names."""
        return self.vector_store.get_document_names()

    def remove_document(self, document_name: str):
        """Remove a document from the store."""
        self.vector_store.delete_document(document_name)
        self.retriever.refresh_bm25_index()

    def clear_all(self):
        """Remove all documents and reset."""
        self.vector_store.clear()
        self.retriever.refresh_bm25_index()
        self.conversation_history = []


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG Pipeline CLI")
    parser.add_argument("--ingest", help="PDF file to ingest")
    parser.add_argument("--query", help="Question to ask")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()

    pipeline = RAGPipeline(config_path=args.config)

    if args.ingest:
        stats = pipeline.ingest_pdf(args.ingest)
        print(f"\nIngested: {stats}")

    if args.query:
        result = pipeline.query(args.query)
        print(f"\nAnswer: {result['answer']}")
        print(f"\nSources:")
        for source in result["sources"]:
            print(f"  - {source['document']}, Page {source['page']}")
