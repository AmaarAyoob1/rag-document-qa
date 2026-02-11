"""Tests for text chunking strategies."""

import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from chunker import chunk_fixed_size, chunk_recursive, chunk_by_sentence, Chunk
from document_loader import DocumentPage


@pytest.fixture
def sample_pages():
    """Create sample document pages for testing."""
    return [
        DocumentPage(
            text="This is the first paragraph of a document. It contains important information. "
                 "The data shows significant growth in Q3 revenue.\n\n"
                 "The second paragraph discusses methodology. We used a randomized control trial "
                 "with 500 participants across three regions.",
            page_number=1,
            document_name="test_doc",
            document_path="/test/test_doc.pdf",
        ),
        DocumentPage(
            text="Results were statistically significant at p < 0.05. The treatment group showed "
                 "a 23% improvement over baseline.",
            page_number=2,
            document_name="test_doc",
            document_path="/test/test_doc.pdf",
        ),
    ]


class TestFixedSizeChunking:
    def test_creates_chunks(self, sample_pages):
        chunks = chunk_fixed_size(sample_pages, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 0
        assert all(isinstance(c, Chunk) for c in chunks)

    def test_respects_chunk_size(self, sample_pages):
        chunks = chunk_fixed_size(sample_pages, chunk_size=100, chunk_overlap=0)
        for chunk in chunks:
            assert len(chunk.text) <= 100

    def test_preserves_metadata(self, sample_pages):
        chunks = chunk_fixed_size(sample_pages, chunk_size=100, chunk_overlap=0)
        for chunk in chunks:
            assert chunk.document_name == "test_doc"
            assert chunk.page_number in [1, 2]

    def test_unique_chunk_ids(self, sample_pages):
        chunks = chunk_fixed_size(sample_pages, chunk_size=100, chunk_overlap=20)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids))


class TestRecursiveChunking:
    def test_creates_chunks(self, sample_pages):
        chunks = chunk_recursive(sample_pages, chunk_size=150, chunk_overlap=20)
        assert len(chunks) > 0

    def test_respects_separators(self, sample_pages):
        """Recursive chunking should prefer paragraph breaks."""
        chunks = chunk_recursive(
            sample_pages, chunk_size=500, chunk_overlap=0,
            separators=["\n\n", "\n", ". ", " "]
        )
        # With 500 char limit, first page should split on paragraph break
        assert len(chunks) >= 2

    def test_preserves_page_numbers(self, sample_pages):
        chunks = chunk_recursive(sample_pages, chunk_size=150, chunk_overlap=20)
        page_numbers = set(c.page_number for c in chunks)
        assert 1 in page_numbers
        assert 2 in page_numbers


class TestSentenceChunking:
    def test_creates_chunks(self, sample_pages):
        chunks = chunk_by_sentence(sample_pages, sentences_per_chunk=2, overlap_sentences=1)
        assert len(chunks) > 0

    def test_source_label(self, sample_pages):
        chunks = chunk_by_sentence(sample_pages, sentences_per_chunk=2)
        for chunk in chunks:
            assert "test_doc" in chunk.source_label
            assert "Page" in chunk.source_label


class TestEdgeCases:
    def test_empty_page(self):
        pages = [DocumentPage(text="", page_number=1, document_name="empty", document_path="/test")]
        chunks = chunk_fixed_size(pages, chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 0

    def test_single_word(self):
        pages = [DocumentPage(text="Hello", page_number=1, document_name="tiny", document_path="/test")]
        chunks = chunk_fixed_size(pages, chunk_size=100, chunk_overlap=0)
        assert len(chunks) == 1
        assert chunks[0].text == "Hello"
