# **N-gram Predictor**

the project is a next-word prediction system using an n-gram language model. the training data is four Sherlock Holmes novels by Arthur Conan Doyle, sourced from Project Gutenberg. The model is trained on all four novels.
At inference time, the system takes the last NGRAM_ORDER − 1 words typed by the user and returns the top-k most probable next words using an n-gram model with backoff to lower orders when the context is unseen.


## **Requirements**

Python=3
nltk=3.8.1
python-dotenv=1.0.0 


## **Setup**

clone the repo; create and activate an Anaconda environment; install dependencies; populate config/.env; download raw .txt files into the correct folders.


### **Project Structure**

```
ngram-predictor/
├── config/
│   └── .env
├── data/
│   ├── raw/
│   │   ├── train/          # Four training books (.txt)
│   │   └── eval/           # One evaluation book (.txt)
│   ├── processed/
│   │   ├── train_tokens.txt
│   │   └── eval_tokens.txt 
│   └── model/
│       ├── model.json      
│       └── vocab.json      
├── src/
│   ├── data_prep/
│   │   └── normalizer.py      # Normalizer class
│   ├── model/
│   │   └── ngram_model.py     # NGramModel class
│   ├── inference/
│   │   └── predictor.py       # Predictor class
│   ├── ui/
│   │   └── app.py             # PredictorUI class          
│   └── evaluation/
│       └── evaluator.py       # Evaluator class            
├── main.py                    # Single entry point — CLI and wiring
├── tests/
│   ├── test_data_prep.py     
│   ├── test_model.py          
│   ├── test_inference.py     
│   ├── test_ui.py             
│   └── test_evaluation.py    
├── .gitignore
├── requirements.txt
└── README.md 

```
---