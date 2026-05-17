import os
import json
import chromadb
from chromadb.utils import embedding_functions
from langchain.text_splitter import RecursiveCharacterTextSplitter

def ingest_to_chroma(data_file="data/dataset.json", db_dir="data/chroma_db"):
    if not os.path.exists(data_file):
        print(f"Error: {data_file} not found. Please run prepare_dataset.py first.")
        return

    print(f"Loading data from {data_file}")
    with open(data_file, 'r', encoding='utf-8') as f:
        documents = json.load(f)
        
    client = chromadb.PersistentClient(path=db_dir)
    
    # Using a fast, local embedding model
    emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    
    collection = client.get_or_create_collection(
        name="hackathon_docs",
        embedding_function=emb_fn
    )
    
    # We need to chunk the text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    
    print("Chunking documents...")
    ids = []
    texts = []
    metadatas = []
    
    for doc in documents:
        chunks = text_splitter.split_text(doc['text'])
        for i, chunk in enumerate(chunks):
            chunk_id = f"{doc['id']}_chunk_{i}"
            ids.append(chunk_id)
            texts.append(chunk)
            metadatas.append({
                "title": doc['title'],
                "url": doc['url'],
                "doc_id": doc['id']
            })
            
    print(f"Total chunks created: {len(texts)}")
    
    # Batch add to chroma to avoid memory issues
    batch_size = 5000
    print("Ingesting into ChromaDB...")
    for i in range(0, len(texts), batch_size):
        end = min(i + batch_size, len(texts))
        collection.add(
            ids=ids[i:end],
            documents=texts[i:end],
            metadatas=metadatas[i:end]
        )
        print(f"Ingested {end}/{len(texts)} chunks...")
        
    print("Basic RAG ingestion complete!")

if __name__ == "__main__":
    ingest_to_chroma()
