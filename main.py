from dotenv import load_dotenv
import os
from pathlib import Path
from src.data_prep.normalizer import Normalizer
# Load environment variables from config/.env
load_dotenv("config/.env")
raw_dir = os.getenv("TRAIN_RAW_DIR")
eval_dir = os.getenv("EVAL_RAW_DIR")
# exits if TRAIN_RAW_DIR is not found in .env
if not raw_dir:
    print("Error: TRAIN_RAW_DIR not found in .env")
    exit()
train_obj = Normalizer(raw_dir)
train_loaded_file = train_obj.load() # Load first
train_normalized_text=train_obj.normalize()
train_sentences = train_obj.sentence_tokenize()
train_removed_punctuation = train_obj.remove_punctuation()
train_words = train_obj.word_tokenize()
train_output_file = train_obj.save(eval_dir + "/train_tokens.txt")

print(train_words[:25]) # Print first 5 words