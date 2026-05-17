import os
import json
try:
    from datasets import load_dataset
except ImportError:
    print("Please install datasets: pip install datasets")
    exit(1)

def estimate_tokens(text: str) -> int:
    """Fast token estimate: ~4 chars per token."""
    return len(text) // 4

def download_and_prepare_dataset(output_dir="data", target_tokens=2_200_000):
    os.makedirs(output_dir, exist_ok=True)
    print("Loading AG News dataset (all 120k samples for 2M+ tokens)...")
    
    # AG News has 120k training samples - load all of them to reach 2M tokens
    # Each article is ~50 tokens, so 120k articles ≈ 6M tokens
    dataset = load_dataset("ag_news", split="train")
    
    total_tokens = 0
    documents = []
    
    print(f"Processing {len(dataset)} articles...")
    for i, row in enumerate(dataset):
        text = row['text']
        tokens = estimate_tokens(text)
        total_tokens += tokens
        documents.append({
            "id": f"ag_news_{i}",
            "url": "N/A",
            "title": f"AG News Article {i} (Label {row['label']})",
            "text": text,
            "tokens": tokens
        })
        if i % 10000 == 0 and i > 0:
            print(f"Processed {i} articles... Total tokens so far: {total_tokens:,}")
        if total_tokens >= target_tokens:
            print(f"Reached {total_tokens:,} tokens at {i+1} articles. Stopping.")
            break
            
    print(f"\nFinal total tokens collected: {total_tokens:,}")
    if total_tokens < 2_000_000:
        print("WARNING: Total tokens is less than the 2M requirement!")
    else:
        print(f"[OK] Dataset meets the 2M+ token requirement ({total_tokens:,} tokens).")
    
    output_file = os.path.join(output_dir, "dataset.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, ensure_ascii=False, indent=2)
        
    print(f"Saved {len(documents)} documents to {output_file}")
    
if __name__ == "__main__":
    download_and_prepare_dataset()
