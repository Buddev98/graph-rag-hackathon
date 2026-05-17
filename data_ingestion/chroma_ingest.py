import os
import json
import chromadb
from chromadb.utils import embedding_functions

def chunk_text(text, chunk_size=600):
    """Simple splitter - for AG News short articles, one chunk per doc is fine."""
    if len(text) <= chunk_size:
        return [text]
    # Split by sentences
    chunks = []
    while len(text) > chunk_size:
        split_at = text.rfind('. ', 0, chunk_size)
        if split_at == -1:
            split_at = chunk_size
        chunks.append(text[:split_at + 1].strip())
        text = text[split_at + 1:].strip()
    if text:
        chunks.append(text)
    return chunks

def ingest_to_chroma(data_file="data/dataset.json", db_dir="data/chroma_db", batch_size=1000):
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run prepare_dataset.py first.")
        return

    print(f"Loading data from {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)

    print(f"Loaded {len(documents)} documents.")

    client = chromadb.PersistentClient(path=db_dir)
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    collection = client.get_or_create_collection(name="hackathon_docs", embedding_function=emb_fn)

    print(f"Ingesting into ChromaDB in batches of {batch_size}...")
    
    ids_batch = []
    texts_batch = []
    metas_batch = []
    total_chunks = 0
    
    def flush_batch():
        nonlocal total_chunks
        if ids_batch:
            collection.add(ids=ids_batch, documents=texts_batch, metadatas=metas_batch)
            total_chunks += len(ids_batch)
            print(f"  Ingested {total_chunks} chunks so far...")
            ids_batch.clear()
            texts_batch.clear()
            metas_batch.clear()

    for doc in documents:
        chunks = chunk_text(doc['text'])
        for i, chunk in enumerate(chunks):
            ids_batch.append(f"{doc['id']}_chunk_{i}")
            texts_batch.append(chunk)
            metas_batch.append({"title": doc['title'], "url": doc['url'], "doc_id": doc['id']})
            if len(ids_batch) >= batch_size:
                flush_batch()

    flush_batch()  # final leftover
    print(f"Basic RAG ingestion complete! Total chunks: {total_chunks}")

if __name__ == "__main__":
    ingest_to_chroma()
