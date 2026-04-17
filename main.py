from dotenv import load_dotenv
import os
from pathlib import Path
from src.data_prep.normalizer import Normalizer
from src.model.ngram_model import NgramModel
from src.inference.predictor import Predictor
# Load environment variables from config/.env
load_dotenv("config/.env")
raw_dir = os.getenv("TRAIN_RAW_DIR")
eval_dir = os.getenv("EVAL_RAW_DIR")
# exits if TRAIN_RAW_DIR is not found in .env
if not raw_dir:
    print("Error: TRAIN_RAW_DIR not found in .env")
    exit()
if not eval_dir:
    print("Error: EVAL_RAW_DIR not found in .env")
    exit()
train_obj = Normalizer(raw_dir)
train_loaded_file = train_obj.load() # Load first
train_normalized_text=train_obj.normalize()
train_sentences = train_obj.sentence_tokenize()
train_removed_punctuation = train_obj.remove_punctuation()
train_words = train_obj.word_tokenize()
train_output_file = train_obj.save(eval_dir + "/train_tokens.txt")
print(train_words[:1]) # Print first tokenized sentences  

Ngram_obj = NgramModel(eval_dir)
Ngram_vocab = Ngram_obj.build_vocab() # Build vocabulary
Ngram_build_counts_and_probabilities = Ngram_obj.build_counts_and_probabilities() # Build counts and probabilities
test_probs = Ngram_obj.lookup(["holmes","said", "to"]) # Example lookup
model_vocab=Ngram_obj.load() # Load the model and vocab
model=model_vocab[0] # Extract model
vocab=model_vocab[1] # Extract vocabulary
print(test_probs) # Print the probabilities for the context

predictor_obj = Predictor(Ngram_obj, train_obj)
input_text = "it came out"
predictions = predictor_obj.predict_next(input_text)
print(f"Predicted next words for '{input_text}': {predictions}")