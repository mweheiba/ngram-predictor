from dotenv import load_dotenv
import os
from pathlib import Path
import json

class NgramModel:
    """A class which processes whole training tokens, building voacbulary,storing, and exposing n-gram probability tables and backoff lookup across all orders from 1 up to NGRAM_ORDER , saving.
    """
    def __init__(self, folder_path):
        """
        Parameters:
        folder_path(string): path of the folder containing tokens training data
        """
        self.folder_path = folder_path + "/train_tokens.txt"
    
    def build_vocab(self, path=None):
        """Builds vocabulary from training tokens."""
        import json
        from collections import Counter
        UNK_THRESHOLD = int(os.getenv('UNK_THRESHOLD', 3))
        word_counts = Counter()
        input_txt = path + "/train_tokens.txt" if path else self.folder_path
        output_json = Path(os.getenv('VOCAB', 'vocab.json'))
        output_json.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists

        # 2. Collect unique words and their counts
        with open(input_txt, 'r', encoding='utf-8') as f:
            for line in f:
                # Assumes word tokens are space-separated
                tokens = line.strip().split()
                word_counts.update(tokens)

        # 3. Filter words by threshold and add <UNK>
        # Only keep words that appear >= threshold times
        vocab = {word for word, count in word_counts.items() if count >= UNK_THRESHOLD}
        vocab.add("<UNK>")

        # 4. Save the sorted vocabulary to .json
        with open(output_json, 'w', encoding='utf-8') as f:
            json.dump(sorted(list(vocab)), f, indent=4)
    
    def build_counts_and_probabilities(self, path=None):
        """Builds n-gram counts and probabilities."""
        from collections import defaultdict
        NGRAM_ORDER = int(os.getenv('NGRAM_ORDER', 4))

        output_file = Path(os.getenv('MODEL', 'model.json'))
        output_file.parent.mkdir(parents=True, exist_ok=True)  # Ensure output directory exists 
        input_file = path + "/test_output.txt" if path else self.folder_path
        counts = {i: defaultdict(lambda: defaultdict(int)) for i in range(1, NGRAM_ORDER + 1)}
    
    # Total tokens for 1-gram probabilities
        total_tokens = 0

        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                tokens = line.strip().split()
                if not tokens:
                    continue
            
                total_tokens += len(tokens)

                for i in range(len(tokens)):
                    for n in range(1, NGRAM_ORDER + 1):
                        if i + n <= len(tokens):
                            ngram = tokens[i:i+n]
                            word = ngram[-1]
                            prefix = " ".join(ngram[:-1])
                        
                            # Store in counts[order][prefix][word]
                            counts[n][prefix][word] += 1

    # Convert counts to MLE Probabilities
        model = {}
        for n in range(1, NGRAM_ORDER + 1):
            order_key = f"{n}gram"
            model[order_key] = {}
        
            for prefix, successors in counts[n].items():
            # For 1-gram, the denominator is the total token count
                if n == 1:
                    for word, count in successors.items():
                        model[order_key][word] = count / total_tokens
                else:
                    # For n > 1, denominator is the sum of all words following this prefix
                    prefix_total = sum(successors.values())
                    model[order_key][prefix] = {
                        word: count / prefix_total 
                        for word, count in successors.items()
                    }

        with open(output_file, 'w', encoding='utf-8') as jf:
            json.dump(model, jf, indent=4)
    
    def lookup(self, context):
        """Looks up the probability of a word given a context using backoff."""
        NGRAM_ORDER = int(os.getenv('NGRAM_ORDER', 4))
        model_file = os.getenv('MODEL', 'model.json')
        vocab_file = os.getenv('VOCAB', 'vocab.json')
        with open(model_file, 'r', encoding='utf-8') as f:
            model = json.load(f)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab = set(json.load(f))
        # 1. Map words not in vocab to <UNK>
        clean_context = [
            word if word in vocab else "<UNK>" 
            for word in context
        ]
        if clean_context[-1] == "<UNK>":
            return {}
        for n in range(NGRAM_ORDER, 1, -1):
            # We need the last (n-1) tokens for an n-gram lookup
            context_size = n - 1
            if len(clean_context) >= context_size:
                # Grab the most recent context tokens
                current_context = " ".join(clean_context[-context_size:])
                order_key = f"{n}gram"
            
                # Check if this specific context exists in our model
                if current_context in model.get(order_key, {}):
                    return model[order_key][current_context]

         # 3. Final Fallback: 1-gram (Unigram)
         # This doesn't depend on context, just the general frequency of words
        if "1gram" in model:
            return model["1gram"]

        return {}
        
    def load(self, model_path=None, vocab_path=None):
        """Loads the model and vocabulary from files."""
        model_file = model_path if model_path else os.getenv('MODEL', 'model.json')
        vocab_file = vocab_path if vocab_path else os.getenv('VOCAB', 'vocab.json')
        with open(model_file, 'r', encoding='utf-8') as f:
            self.model = json.load(f)
        with open(vocab_file, 'r', encoding='utf-8') as f:
            self.vocab = set(json.load(f))
        return self.model, self.vocab

def main():
    # Load environment variables from config/.env
    load_dotenv("config/.env")
    eval_dir = os.getenv("EVAL_RAW_DIR")
    # exits if TRAIN_RAW_DIR is not found in .env
    if not eval_dir:
        print("Error: EVAL_RAW_DIR not found in .env")
        return
    test_file_obj = NgramModel(eval_dir)
    test_build_vocab = test_file_obj.build_vocab() # Build vocabulary
    test_build_counts_and_probabilities = test_file_obj.build_counts_and_probabilities() # Build counts and probabilities
    test_probs = test_file_obj.lookup(["i","play", "work"]) # Example lookup
    #print(test_probs) # Print the probabilities for the context
    model_vocab=test_file_obj.load() # Load the model and vocab
    model=model_vocab[0] # Extract model
    #print(model) # Print the loaded model and vocab
    vocab=model_vocab[1] # Extract vocabulary
    #print(vocab) # Print the loaded vocabulary

           
    

    
        
if __name__ == '__main__':
    main()