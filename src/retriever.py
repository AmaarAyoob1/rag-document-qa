"""
Hybrid retriever combining semantic search, BM25 keyword search,
Reciprocal Rank Fusion, and cross-encoder reranking.
"""

import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from embeddings import EmbeddingModel
from vector_store import VectorStore


class HybridRetriever:
    def __init__(self, embedding_model, vector_store, use_hybrid=True,
                 semantic_weight=0.7, bm25_weight=0.3,
                 reranker_model="cross-encoder/ms-marco-MiniLM-L-6-v2", **kwargs):
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.use_hybrid = use_hybrid
        self.semantic_weight = semantic_weight
        self.bm25_weight = bm25_weight

        # BM25 index
        self.bm25 = None
        self.bm25_chunks = []

        # Cross-encoder reranker
        if reranker_model and reranker_model != "none":
            print(f"Loading reranker: {reranker_model}")
            self.reranker = CrossEncoder(reranker_model)
        else:
            self.reranker = None

        print(f"Retriever initialized (hybrid={use_hybrid}, reranker={'yes' if self.reranker else 'no'})")

    def refresh_bm25_index(self):
        """Rebuild BM25 index from all chunks in vector store."""
        all_data = self.vector_store.get_all()
        if not all_data or not all_data.get("documents"):
            return

        self.bm25_chunks = []
        tokenized_corpus = []

        for i, doc_text in enumerate(all_data["documents"]):
            metadata = all_data["metadatas"][i] if all_data.get("metadatas") else {}
            chunk_id = all_data["ids"][i] if all_data.get("ids") else str(i)

            self.bm25_chunks.append({
                "id": chunk_id,
                "text": doc_text,
                "metadata": metadata
            })
            tokenized_corpus.append(doc_text.lower().split())

        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            print(f"BM25 index built with {len(tokenized_corpus)} chunks")

    def _semantic_search(self, query, top_k=10, filter_doc=None):
        """Vector similarity search via ChromaDB."""
        query_embedding = self.embedding_model.embed_query(query)
        results = self.vector_store.query(query_embedding, top_k=top_k, filter_doc=filter_doc)
        return results

    def _bm25_search(self, query, top_k=10):
        """Keyword search using BM25."""
        if self.bm25 is None or not self.bm25_chunks:
            return []

        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                chunk = self.bm25_chunks[idx]
                results.append({
                    "id": chunk["id"],
                    "text": chunk["text"],
                    "metadata": chunk["metadata"],
                    "score": float(scores[idx])
                })
        return results

    def _reciprocal_rank_fusion(self, semantic_results, bm25_results, k=60):
        """Merge results from semantic and BM25 search using RRF."""
        doc_scores = {}
        doc_data = {}

        for rank, result in enumerate(semantic_results):
            doc_id = result["id"]
            rrf_score = self.semantic_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            doc_data[doc_id] = result

        for rank, result in enumerate(bm25_results):
            doc_id = result["id"]
            rrf_score = self.bm25_weight / (k + rank + 1)
            doc_scores[doc_id] = doc_scores.get(doc_id, 0) + rrf_score
            if doc_id not in doc_data:
                doc_data[doc_id] = result

        sorted_ids = sorted(doc_scores, key=doc_scores.get, reverse=True)

        merged = []
        for doc_id in sorted_ids:
            result = doc_data[doc_id]
            result["rrf_score"] = doc_scores[doc_id]
            merged.append(result)

        return merged

    def _rerank(self, query, results, top_k=5):
        """Rerank results using cross-encoder."""
        if not self.reranker or not results:
            return results[:top_k]

        pairs = [[query, r["text"]] for r in results]
        scores = self.reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)

        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]

    def retrieve(self, query, top_k=10, rerank_top_k=5, filter_doc=None):
        """Full retrieval pipeline: semantic + BM25 + RRF + rerank."""
        # Step 1: Semantic search
        semantic_results = self._semantic_search(query, top_k=top_k, filter_doc=filter_doc)

        if self.use_hybrid and self.bm25 is not None:
            # Step 2: BM25 search
            bm25_results = self._bm25_search(query, top_k=top_k)

            # Step 3: Reciprocal Rank Fusion
            merged = self._reciprocal_rank_fusion(semantic_results, bm25_results)
        else:
            merged = semantic_results

        # Step 4: Rerank with cross-encoder
        final = self._rerank(query, merged, top_k=rerank_top_k)

        return final
