"""
Hybrid retrieval combining semantic search and BM25 keyword search.
Includes cross-encoder reranking for improved precision.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from embeddings import EmbeddingModel
from vector_store import VectorStore


class HybridRetriever:
    """
    Combines vector similarity search with BM25 keyword matching.
    
    Why hybrid? Semantic search finds paraphrases and conceptual matches,
    but can miss exact keyword matches. BM25 catches those. Together they
    cover more ground than either alone.
    """

    def __init__(
        self,
        embedding_model: EmbeddingModel,
        vector_store: VectorStore,
        use_hybrid: bool = True,
        semantic_weight: float = 0.7,
        bm25_weight: float = 0.3,
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # Initialize BM25 index from all stored documents
        self._build_bm25_index()

        # Initialize cross-encoder reranker
        self.reranker = CrossEncoder(reranker_model)
        print(f"Retriever initialized (hybrid={use_hybrid})")

    def _build_bm25_index(self):
        """Build BM25 index from all chunks in the vector store."""
        results = self.vector_store.collection.get(
            include=["documents", "metadatas"]
        )

        self.bm25_docs = results["documents"] or []
        self.bm25_metadatas = results["metadatas"] or []
        self.bm25_ids = results["ids"] or []

        if self.bm25_docs:
            tokenized = [doc.lower().split() for doc in self.bm25_docs]
            self.bm25 = BM25Okapi(tokenized)
        else:
            self.bm25 = None

    def refresh_bm25_index(self):
        """Rebuild BM25 index (call after adding new documents)."""
        self._build_bm25_index()

    def retrieve(
        self,
        query: str,
        top_k: int = 10,
        rerank_top_k: int = 5,
        filter_doc: str = None,
    ) -> list[dict]:
        """
        Retrieve relevant chunks for a query.
        
        Pipeline:
        1. Semantic search via vector store
        2. BM25 keyword search (if hybrid enabled)
        3. Reciprocal Rank Fusion to merge results
        4. Cross-encoder reranking for final precision
        
        Args:
            query: User question
            top_k: Number of candidates to retrieve
            rerank_top_k: Number of final results after reranking
            filter_doc: Optional document name filter
        
        Returns:
            List of dicts with 'text', 'metadata', 'score'
        """
        # Step 1: Semantic search
        query_embedding = self.embedding_model.embed_query(query)
        semantic_results = self.vector_store.query(
            query_embedding, top_k=top_k, filter_doc=filter_doc
        )

        if not self.use_hybrid or self.bm25 is None:
            candidates = semantic_results
        else:
            # Step 2: BM25 search
            bm25_results = self._bm25_search(query, top_k=top_k)

            # Step 3: Reciprocal Rank Fusion
            candidates = self._reciprocal_rank_fusion(
                semantic_results, bm25_results
            )

        if not candidates:
            return []

        # Step 4: Rerank with cross-encoder
        reranked = self._rerank(query, candidates, top_k=rerank_top_k)

        return reranked

    def _bm25_search(self, query: str, top_k: int = 10) -> list[dict]:
        """Search using BM25 keyword matching."""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)

        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "id": self.bm25_ids[idx],
                    "text": self.bm25_docs[idx],
                    "metadata": self.bm25_metadatas[idx],
                    "score": float(scores[idx]),
                })

        return results

    def _reciprocal_rank_fusion(
        self,
        semantic_results: list[dict],
        bm25_results: list[dict],
        k: int = 60,
    ) -> list[dict]:
        """
        Merge results from semantic and BM25 search using RRF.
        
        RRF score = sum(1 / (k + rank_i)) for each result list.
        This is robust to different score scales between retrieval methods.
        """
        doc_scores = {}
        doc_data = {}

        # Score semantic results
        for rank, result in enumerate(semantic_results):
            doc_id = result["id"]
            rrf_score = self.semantic_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = result

        # Score BM25 results
        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = self.bm25_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        # Sort by fused score
        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

        fused_results = []
        for doc_id in sorted_ids:
            result = doc_data[doc_id].copy()
            result["score"] = doc_scores[doc_id]
            fused_results.append(result)

        return fused_results

    def _rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        """
        Rerank candidates using a cross-encoder model.
        
        Cross-encoders are more accurate than bi-encoders for relevance
        scoring, but too slow for initial retrieval. Using them as a
        second stage on a small candidate set gives the best of both worlds.
        """
        if not candidates:
            return []

        # Create query-document pairs for the cross-encoder
        pairs = [(query, cand["text"]) for cand in candidates]

        # Score all pairs
        scores = self.reranker.predict(pairs)

        # Attach scores and sort
        for i, cand in enumerate(candidates):
            cand["rerank_score"] = float(scores[i])

        reranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)

        return reranked[:top_k]
