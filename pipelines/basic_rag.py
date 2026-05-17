import os
import chromadb
from chromadb.utils import embedding_functions
import google.generativeai as genai

def run_basic_rag(query: str):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return {"response": "Error: GEMINI_API_KEY missing", "tokens": 0, "cost": 0.0, "accuracy": "N/A"}
            
        genai.configure(api_key=api_key)
        
        # Connect to Chroma
        db_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "chroma_db")
        if not os.path.exists(db_dir):
            return {"response": "Error: ChromaDB not found. Please run chroma_ingest.py first.", "tokens": 0, "cost": 0.0, "accuracy": "N/A"}
            
        client = chromadb.PersistentClient(path=db_dir)
        emb_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        
        try:
            collection = client.get_collection(name="hackathon_docs", embedding_function=emb_fn)
        except Exception:
            return {"response": "Error: Chroma collection 'hackathon_docs' not found.", "tokens": 0, "cost": 0.0, "accuracy": "N/A"}
        
        # Retrieve
        results = collection.query(
            query_texts=[query],
            n_results=5
        )
        
        context_chunks = results['documents'][0] if results['documents'] else []
        context = "\n\n---\n\n".join(context_chunks)
        
        # Prompt
        prompt = f"""You are a helpful assistant. Use the following retrieved context to answer the user's question. If you don't know the answer based on the context, say so.

Context:
{context}

Question:
{query}

Answer:"""

        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata'):
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        elif hasattr(response, '_usage_metadata'):
            prompt_tokens = response._usage_metadata.prompt_token_count
            completion_tokens = response._usage_metadata.candidates_token_count
            
        total_tokens = prompt_tokens + completion_tokens
        cost = (prompt_tokens / 1_000_000 * 0.075) + (completion_tokens / 1_000_000 * 0.30)
        
        return {
            "response": response.text,
            "tokens": total_tokens,
            "cost": cost,
            "accuracy": "Pending" 
        }
    except Exception as e:
        return {"response": f"Error: {str(e)}", "tokens": 0, "cost": 0.0, "accuracy": "N/A"}
