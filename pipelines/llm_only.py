import os
import google.generativeai as genai

def run_llm_only(query: str):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key or api_key == "your_gemini_api_key_here":
            return {
                "response": "Error: Please set a valid GEMINI_API_KEY in .env",
                "tokens": 0,
                "cost": 0.0,
                "accuracy": "N/A"
            }
            
        genai.configure(api_key=api_key)
        
        model = genai.GenerativeModel('gemini-2.0-flash')
        response = model.generate_content(query)
        
        prompt_tokens = 0
        completion_tokens = 0
        if hasattr(response, 'usage_metadata'):
            prompt_tokens = getattr(response.usage_metadata, 'prompt_token_count', 0)
            completion_tokens = getattr(response.usage_metadata, 'candidates_token_count', 0)
        elif hasattr(response, '_usage_metadata'):
            prompt_tokens = response._usage_metadata.prompt_token_count
            completion_tokens = response._usage_metadata.candidates_token_count
            
        total_tokens = prompt_tokens + completion_tokens
        # Cost estimate based on gemini-1.5-flash
        cost = (prompt_tokens / 1_000_000 * 0.075) + (completion_tokens / 1_000_000 * 0.30)
        
        return {
            "response": response.text,
            "tokens": total_tokens,
            "cost": cost,
            "accuracy": "Pending" 
        }
    except Exception as e:
        return {
            "response": f"Error: {str(e)}",
            "tokens": 0,
            "cost": 0.0,
            "accuracy": "N/A"
        }
