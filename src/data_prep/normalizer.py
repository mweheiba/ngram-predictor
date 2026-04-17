from dotenv import load_dotenv
import os
from pathlib import Path

class Normalizer:
    """A class which processes whole raw files loading, stripping, cleaning, 
    tokenizing, and saving.
    """
    def __init__(self, folder_path):
        """
        Parameters:
        folder_path(string): path of the folder containing files to be normalized
        """
        self.folder_path = folder_path+"/*.txt"
        self.text_file = ""
        self.sentences = []
        self.words = []

    def load(self, path=None):
        """Loads file content."""
        import glob
        target_path = path + "/*.txt" if path else self.folder_path
        for file_path in glob.glob(target_path):
            with open(file_path, 'r', encoding='utf-8') as file:
                read_file = file.read()
                read_file = self.strip_gutenberg(read_file)
                self.text_file += read_file + "\n"  # Add newline between files
        return self.text_file
        
    def strip_gutenberg(self, text=None):
        """
        Removes Gutenberg header and footer.
        Returns: string: striped string.
        """
        import re
        text = self.text_file if text is None else text
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*"
        start_match = re.search(start_pattern, text, re.S)
        end_match = re.search(end_pattern, text, re.S)
        if start_match:
            stripped_text = text[start_match.end(): end_match.start()] if end_match else text[start_match.end():]
        else:
            stripped_text = text
        return stripped_text
    
    def lowercase(self):
        """Converts text to lowercase."""
        self.text_file = self.text_file.lower()
        return self.text_file
    
    def remove_punctuation(self):
        """Removes punctuation from text."""
        import string
        translator = str.maketrans('', '', string.punctuation)
        self.sentences = [sentence.translate(translator) for sentence in self.sentences]
        self.sentences = [sentence.encode("ascii", "ignore").decode("ascii") for sentence in self.sentences]
        return self.sentences
    
    def remove_numbers(self):
        """Removes numbers from text."""
        self.text_file = ''.join([char for char in self.text_file if not char.isdigit()])
        return self.text_file
    
    def remove_whitespace(self):
        """Removes extra whitespace from text."""
        self.text_file = ' '.join(self.text_file.split())
        return self.text_file
    
    def remove_blank_lines(self):
        """Removes blank lines from text."""
        self.text_file = '\n'.join([line for line in self.text_file.splitlines() if line.strip()])
        return self.text_file
    
    def normalize(self):
        """Runs all normalization steps in sequence."""
        self.lowercase()
        self.remove_numbers()
        #self.remove_punctuation()
        self.remove_whitespace()
        self.remove_blank_lines()
        return self.text_file
    
    def sentence_tokenize(self):
        """Tokenizes text into sentences."""
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
        from nltk.tokenize import sent_tokenize
        self.sentences = sent_tokenize(self.text_file)
        return self.sentences
    
    def word_tokenize(self):
        """Tokenizes text into words."""
        import nltk
        nltk.download('punkt')
        nltk.download('punkt_tab')
        from nltk.tokenize import word_tokenize
        self.words = [word_tokenize(sentence) for sentence in self.sentences]
        return self.words
    
    def save(self, output_path):
        """Saves tokenized sententences to a file."""
        with open(output_path, 'w', encoding='utf-8') as file:
            file.write('\n'.join([' '.join(words) for words in self.words]))

def main():
        # Load environment variables from config/.env
        load_dotenv("config/.env")
        raw_dir = os.getenv("TRAIN_RAW_DIR")
        eval_dir = os.getenv("EVAL_RAW_DIR")
        # exits if TRAIN_RAW_DIR is not found in .env
        if not raw_dir:
            print("Error: TRAIN_RAW_DIR not found in .env")
            return
        test_file_obj = Normalizer(raw_dir)
        test_loaded_file = test_file_obj.load(raw_dir) # Load first
        test_normalized_text=test_file_obj.normalize()
        test_sentences = test_file_obj.sentence_tokenize()
        test_removed_punctuation = test_file_obj.remove_punctuation()
        test_words = test_file_obj.word_tokenize()
        test_output_file = test_file_obj.save(eval_dir + "/train_tokens.txt")
               
        #print(test_normalized_text[:500]) # Print first 500 chars of normalized text
        #print(test_loaded_file[:500]) # Print first 500 chars of loaded text
        #print(test_sentences[:5]) # Print first 5 sentences
        print(test_words[:25]) # Print first 5 words

    
        
if __name__ == '__main__':
        main()