import os
from pathlib import Path
from uuid import uuid4

from app.ingestion.loader import load_document
from app.ingestion.cleaner import clean_pages
from app.ingestion.chunker import chunk_pages
from app.embeddings.embedder import embed_texts
from app.vectorstore.chroma_store import add_chunks, get_collection

def ingest_corpus():
    docs_dir = Path("data/sample_documents")
    files_to_index = [
        f for f in docs_dir.glob("*.*")
        if f.suffix.lower() in [".pdf", ".md", ".txt"] and f.name.lower() != "readme.md"
    ]

    print(f"Found {len(files_to_index)} support documents to ingest:")
    for f in files_to_index:
        print(f" - {f.name}")

    total_chunks = 0
    for doc_path in files_to_index:
        doc_id = uuid4().hex[:12]
        pages = load_document(doc_path)
        cleaned = clean_pages(pages)
        chunks = chunk_pages(cleaned)

        if not chunks:
            print(f"Skipping {doc_path.name} (no text found)")
            continue

        for i, ch in enumerate(chunks, 1):
            ch["chunk_id"] = f"{doc_id}_{ch['page_number'] or 1}_{i}"
            ch["metadata"] = {
                "document_id": doc_id,
                "filename": doc_path.name,
                "document_name": doc_path.name.replace("_", " ").replace(".md", "").replace(".pdf", "").title(),
                "page_number": ch["page_number"] or 1,
                "chunk_number": i,
                "source": doc_path.name,
            }

        texts = [ch["text"] for ch in chunks]
        embeddings = embed_texts(texts)
        count = add_chunks(chunks, embeddings)
        total_chunks += count
        print(f"Indexed {doc_path.name}: {count} chunks")

    col = get_collection()
    print(f"\nTotal collection size in Chroma: {col.count()} chunks.")
    print("Support knowledge base ingestion complete!")

if __name__ == "__main__":
    ingest_corpus()
