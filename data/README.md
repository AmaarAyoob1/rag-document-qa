# Data

## How It Works

This project doesn't ship with data — **you bring your own documents**.

Upload any PDF through the Streamlit interface and the pipeline will:
1. Extract text (preserving page numbers)
2. Chunk it into overlapping segments
3. Embed and store in ChromaDB

## Testing

For quick testing, try uploading any freely available PDF:
- [Attention Is All You Need (Vaswani et al.)](https://arxiv.org/pdf/1706.03762) — the transformer paper
- Any public company 10-K filing from [SEC EDGAR](https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&type=10-K)
- Your own lecture notes, research papers, or reports

## Directory Structure

```
data/
├── README.md           ← You are here
├── vectorstore/        ← ChromaDB storage (auto-created, gitignored)
└── uploads/            ← Temporary uploaded files (gitignored)
```

## Notes

- No data files are committed to this repo
- Vector store is persisted locally and recreated when you ingest documents
- To start fresh, delete `data/vectorstore/` or use the "Clear All" button in the app
