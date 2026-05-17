import os
import json
try:
    from datasets import load_dataset
    from transformers import AutoTokenizer
except ImportError:
    print("Please install datasets and transformers: pip install datasets transformers")
    exit(1)

def download_and_prepare_dataset(output_dir="data", num_articles=2500):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading Wikipedia dataset snippet...")
    
    # trust_remote_code is needed for some HF datasets
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_articles}]", trust_remote_code=True)
    
    # We use gpt2 tokenizer just for a rough token count estimate
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    total_tokens = 0
    documents = []
    
    print(f"Processing {num_articles} articles...")
    for i, row in enumerate(dataset):
        text = row['text']
        tokens = len(tokenizer.encode(text, truncation=False))
        total_tokens += tokens
        documents.append({
            "id": row['id'],
            "url": row['url'],
            "title": row['title'],
            "text": text,
            "tokens": tokens
        })
        if i % 500 == 0 and i > 0:
            print(f"Processed {i} articles... Total tokens so far: {total_tokens}")
            
    print(f"Final total tokens collected: {total_tokens}")
    if total_tokens < 2_000_000:
        print("WARNING: Total tokens is less than the 2M requirement!")
    
    output_file = os.path.join(output_dir, "dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(documents)} documents to {output_file}")
    
if __name__ == "__main__":
    download_and_prepare_dataset()
