from dotenv import load_dotenv
import os
from pathlib import Path
import json
import sys

# Add the project root to the path to allow imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

class Predictor:
    """accepting a pre-loaded NGramModel and Normalizer via the constructor, normalizing input text, and returning the top-k predicted next words sorted by probability. Backoff lookup is delegated to NGramModel.lookup()
    """
    def __init__(self, ngram_model_instance, normalizer_instance):
        """
        Parameters:
        model(NGramModel): pre-loaded n-gram model
        normalizer(Normalizer): pre-loaded text normalizer
        """
        self.ngram_model = ngram_model_instance
        self.normalizer = normalizer_instance
        self.context = []
        
    def normalize(self, text):
        """Normalizes input text using the provided normalizer."""
        NGRAM_ORDER = int(os.getenv('NGRAM_ORDER', 4))
        max_context_size = NGRAM_ORDER - 1
        # Normalize the input text by temporarily setting the normalizer's text_file
        self.normalizer.text_file = text
        normalized_text = self.normalizer.normalize()
        self.normalizer.remove_punctuation()
        normalized_text = self.normalizer.text_file
        tokens = normalized_text.split()
        self.context = tokens[-max_context_size:] if len(tokens) >= max_context_size else tokens
        return self.context


    def predict_next(self, text, top_k=3):
        """Predicts the next words based on the input text."""
        top_k = int(os.getenv('TOP_K', top_k))
        # Normalize the input text
        normalized_context = self.normalize(text)
        predictions_dict = self.ngram_model.lookup(normalized_context)
        # 4. Logic check: if lookup returned an empty dict, return an empty list
        if not predictions_dict:
            return []
        
        # 5. Sort the dictionary by probability (the value) in descending order
        # Result is a list of tuples: [('word', 0.42), ('word2', 0.12)...]
        sorted_predictions = sorted(
            predictions_dict.items(), 
            key=lambda item: item[1], 
            reverse=True
        )
        
        # 6. Return only the top k  words
        top_k_words = []
        for word, prob in sorted_predictions[:top_k]:
            top_k_words.append(word)
        return top_k_words

def main():
    # Example usage
    from src.data_prep.normalizer import Normalizer
    from src.model.ngram_model import NgramModel
    load_dotenv("config/.env")
    eval_dir = os.getenv("EVAL_RAW_DIR")
    raw_dir = os.getenv("TRAIN_RAW_DIR")
    
    if not eval_dir:
        print("Error: EVAL_RAW_DIR not found in .env")
        return
    
    if not raw_dir:
        print("Error: TRAIN_RAW_DIR not found in .env")
        return
    
    normalizer_instance = Normalizer(raw_dir)
    ngram_model_instance = NgramModel(eval_dir)
    predictor = Predictor(ngram_model_instance, normalizer_instance)
    
    input_text = "it came out"
    predictions = predictor.predict_next(input_text)
    print(f"Predicted next words for '{input_text}': {predictions}")
  
        
if __name__ == '__main__':
    main()