import os
import torch

class AnswerEvaluator:
    def __init__(self):
        try:
            import evaluate
            self.bertscore = evaluate.load("bertscore")
        except Exception as e:
            print(f"Warning: Could not load BERTScore. {e}")
            self.bertscore = None
            
    def evaluate_bertscore(self, predictions, references):
        if not self.bertscore:
            return {"f1": [0.0]}
            
        results = self.bertscore.compute(predictions=predictions, references=references, lang="en")
        return results
        
    def evaluate_llm_judge(self, question, answer):
        # A simple placeholder. The hackathon rules specify a HuggingFace PASS/FAIL model.
        # This will be updated to hit the required HF endpoint later.
        if len(answer) > 20 and "Error" not in answer:
            return "PASS"
        return "FAIL"

# Singleton instance
evaluator = AnswerEvaluator()
