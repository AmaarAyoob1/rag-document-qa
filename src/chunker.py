"""
Text chunking strategies for RAG.
Splits documents into overlapping chunks while preserving metadata.
"""

from dataclasses import dataclass, field

from document_loader import Document, DocumentPage


@dataclass
class Chunk:
    """A text chunk with full provenance metadata for citations."""
    text: str
    chunk_id: str
    document_name: str
    page_number: int
    start_char: int
    end_char: int
    metadata: dict = field(default_factory=dict)

    @property
    def source_label(self) -> str:
        return f"{self.document_name}, Page {self.page_number}"


def chunk_fixed_size(
    pages: list[DocumentPage],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[Chunk]:
    """
    Simple fixed-size chunking with overlap.
    
    Splits text into chunks of exactly chunk_size characters
    with chunk_overlap characters of overlap between consecutive chunks.
    """
    chunks = []
    chunk_idx = 0

    for page in pages:
        text = page.text
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{page.document_name}_p{page.page_number}_c{chunk_idx}",
                    document_name=page.document_name,
                    page_number=page.page_number,
                    start_char=start,
                    end_char=end,
                ))
                chunk_idx += 1

            start += chunk_size - chunk_overlap

    return chunks


def chunk_recursive(
    pages: list[DocumentPage],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    separators: list[str] = None,
) -> list[Chunk]:
    """
    Recursive character text splitting.
    
    Tries to split on the most semantically meaningful separator first
    (paragraph breaks > newlines > sentences > words > characters).
    This produces more coherent chunks than fixed-size splitting.
    """
    if separators is None:
        separators = ["\n\n", "\n", ". ", " ", ""]

    chunks = []
    chunk_idx = 0

    for page in pages:
        text = page.text
        page_chunks = _recursive_split(text, chunk_size, chunk_overlap, separators)

        current_pos = 0
        for chunk_text in page_chunks:
            chunk_text = chunk_text.strip()
            if not chunk_text:
                continue

            # Find approximate position in original text
            pos = text.find(chunk_text[:50], current_pos)
            if pos == -1:
                pos = current_pos

            chunks.append(Chunk(
                text=chunk_text,
                chunk_id=f"{page.document_name}_p{page.page_number}_c{chunk_idx}",
                document_name=page.document_name,
                page_number=page.page_number,
                start_char=pos,
                end_char=pos + len(chunk_text),
            ))
            chunk_idx += 1
            current_pos = pos + len(chunk_text) - chunk_overlap

    return chunks


def _recursive_split(
    text: str,
    chunk_size: int,
    chunk_overlap: int,
    separators: list[str],
) -> list[str]:
    """Recursively split text using a hierarchy of separators."""
    if len(text) <= chunk_size:
        return [text] if text.strip() else []

    # Find the best separator that actually exists in the text
    separator = ""
    for sep in separators:
        if sep in text:
            separator = sep
            break

    # Split on the chosen separator
    if separator:
        splits = text.split(separator)
    else:
        # Last resort: character-level split
        splits = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - chunk_overlap)]
        return splits

    # Merge splits into chunks that fit within chunk_size
    chunks = []
    current_chunk = ""

    for split in splits:
        # If adding this split would exceed chunk_size, save current and start new
        test_chunk = current_chunk + separator + split if current_chunk else split

        if len(test_chunk) <= chunk_size:
            current_chunk = test_chunk
        else:
            if current_chunk:
                chunks.append(current_chunk)

            # If single split exceeds chunk_size, recurse with next separator
            if len(split) > chunk_size:
                remaining_separators = separators[separators.index(separator) + 1:]
                if remaining_separators:
                    sub_chunks = _recursive_split(split, chunk_size, chunk_overlap, remaining_separators)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(split[:chunk_size])
                current_chunk = ""
            else:
                current_chunk = split

    if current_chunk:
        chunks.append(current_chunk)

    # Add overlap between chunks
    if chunk_overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev_tail = chunks[i - 1][-chunk_overlap:]
            overlapped.append(prev_tail + " " + chunks[i])
        chunks = overlapped

    return chunks


def chunk_by_sentence(
    pages: list[DocumentPage],
    sentences_per_chunk: int = 5,
    overlap_sentences: int = 1,
) -> list[Chunk]:
    """
    Sentence-based chunking.
    Groups N sentences together with overlap.
    """
    import re
    chunks = []
    chunk_idx = 0

    for page in pages:
        # Simple sentence splitting
        sentences = re.split(r'(?<=[.!?])\s+', page.text)
        sentences = [s.strip() for s in sentences if s.strip()]

        for i in range(0, len(sentences), sentences_per_chunk - overlap_sentences):
            chunk_sentences = sentences[i:i + sentences_per_chunk]
            chunk_text = " ".join(chunk_sentences)

            if chunk_text:
                chunks.append(Chunk(
                    text=chunk_text,
                    chunk_id=f"{page.document_name}_p{page.page_number}_c{chunk_idx}",
                    document_name=page.document_name,
                    page_number=page.page_number,
                    start_char=0,
                    end_char=len(chunk_text),
                ))
                chunk_idx += 1

    return chunks


def chunk_document(
    document: Document,
    strategy: str = "recursive",
    chunk_size: int = 512,
    chunk_overlap: int = 50,
    **kwargs,
) -> list[Chunk]:
    """
    Chunk a document using the specified strategy.
    
    Args:
        document: Parsed Document object
        strategy: One of "fixed", "sentence", "recursive"
        chunk_size: Target chunk size in characters
        chunk_overlap: Overlap between consecutive chunks
    
    Returns:
        List of Chunk objects with metadata
    """
    if strategy == "fixed":
        chunks = chunk_fixed_size(document.pages, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        chunks = chunk_by_sentence(document.pages, **kwargs)
    elif strategy == "recursive":
        chunks = chunk_recursive(document.pages, chunk_size, chunk_overlap)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")

    print(f"Chunked '{document.name}': {len(chunks)} chunks (strategy={strategy}, size={chunk_size})")
    return chunks
