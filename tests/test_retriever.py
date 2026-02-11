"""Tests for retrieval and ranking logic."""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestReciprocalRankFusion:
    """Test RRF merge logic without needing full retriever initialization."""

    def _rrf_merge(self, list_a, list_b, weight_a=0.7, weight_b=0.3, k=60):
        """Standalone RRF for testing."""
        scores = {}
        data = {}

        for rank, item in enumerate(list_a):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + weight_a / (k + rank + 1)
            data[doc_id] = item

        for rank, item in enumerate(list_b):
            doc_id = item["id"]
            scores[doc_id] = scores.get(doc_id, 0) + weight_b / (k + rank + 1)
            if doc_id not in data:
                data[doc_id] = item

        sorted_ids = sorted(scores, key=scores.get, reverse=True)
        return [{"id": did, "score": scores[did], **data[did]} for did in sorted_ids]

    def test_merges_unique_results(self):
        list_a = [{"id": "a", "text": "doc a"}, {"id": "b", "text": "doc b"}]
        list_b = [{"id": "c", "text": "doc c"}, {"id": "d", "text": "doc d"}]

        merged = self._rrf_merge(list_a, list_b)
        ids = [r["id"] for r in merged]
        assert set(ids) == {"a", "b", "c", "d"}

    def test_shared_results_rank_higher(self):
        """Items appearing in both lists should get boosted."""
        list_a = [{"id": "shared", "text": "x"}, {"id": "a_only", "text": "y"}]
        list_b = [{"id": "shared", "text": "x"}, {"id": "b_only", "text": "z"}]

        merged = self._rrf_merge(list_a, list_b)
        assert merged[0]["id"] == "shared"

    def test_empty_lists(self):
        merged = self._rrf_merge([], [])
        assert merged == []

    def test_one_empty_list(self):
        list_a = [{"id": "a", "text": "doc a"}]
        merged = self._rrf_merge(list_a, [])
        assert len(merged) == 1
        assert merged[0]["id"] == "a"
