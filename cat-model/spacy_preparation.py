import spacy
from spacy.tokens import DocBin
import json
import random
import logging
import string
import re
from spacy.lang.pt.stop_words import STOP_WORDS

# Setup logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s', filename='categorization_preparation.log', filemode='w')
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

nlp = spacy.load("pt_core_news_lg", disable=["ner", "parser", "tagger"])
train_db = DocBin()  # DocBin for training data
dev_db = DocBin()    # DocBin for development data

# Define static patterns
static_patterns = {
    'EMAIL': re.compile(r'\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b', re.IGNORECASE),
    'URL': re.compile(r'\b(?:https?://|www\.)\S+?\.br(?=\b|/|\s|[?.])', re.IGNORECASE),
    'DATE': re.compile(r'\b(\d{1,2} de [a-zA-ZçÇ]+ de \d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\b', re.IGNORECASE),
    'UASG': re.compile(r'\bUASG(?:\s+[Nnº°.:]*)?[\s:]*?(\d+(?:[/-]\d+)*)', re.IGNORECASE)
}

def preprocessing(text):
    """Preprocesses the text by applying various transformations."""
    text = text.lower()
    
    # Replace static patterns with a space
    for pattern_name, pattern in static_patterns.items():
        text = pattern.sub(' ', text)
    
    # Tokenize text
    tokens = [token.text for token in nlp(text)]
    
    # Filter tokens
    tokens = [t for t in tokens if 
              t not in STOP_WORDS and 
              t not in string.punctuation and 
              len(t) > 3]
    
    # Remove digits
    tokens = [t for t in tokens if not t.isdigit()]
    
    return " ".join(tokens)

def load_data(file_path, limit=None):
    """Loads data from a JSONL file and optionally limits the number of lines."""
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data_lines = file.readlines()
            logging.info("Loaded data from file successfully.")
        if limit:
            data_lines = data_lines[:limit]
        return data_lines
    except FileNotFoundError:
        logging.error("File not found. Please check the file path and try again.")
        exit()

def split_data(data_lines, train_ratio=0.8, test_ratio=0.1):
    """Splits data into training, development, and test sets based on specified ratios."""
    if train_ratio + test_ratio > 1:
        raise ValueError("Sum of train_ratio and test_ratio must not exceed 1.")
    
    random.shuffle(data_lines)
    
    train_index = int(len(data_lines) * train_ratio)
    test_index = int(len(data_lines) * (1 - test_ratio))
    
    training_data = data_lines[:train_index]
    test_data = data_lines[test_index:]
    development_data = data_lines[train_index:test_index]
    
    logging.info(f"Training Data: {len(training_data)} entries")
    logging.info(f"Development Data: {len(development_data)} entries")
    logging.info(f"Test Data: {len(test_data)} entries")
    
    return training_data, development_data, test_data

def process_text(text, label):
    """Processes text, applies preprocessing, and applies the label to the document."""
    preprocessed_text = preprocessing(text)
    doc = nlp(preprocessed_text)

    categories = [
        "Portaria",
        "Extrato de Contrato",
        "Extrato de Convênio",
        "Edital",
        "Aviso de Licitação",
        "Resultado de Julgamento",
        "Extrato de Termo Aditivo"
    ]

    # Initialize all categories to 0
    doc.cats = {category: 0 for category in categories}

    # Set the relevant category to 1
    if label in doc.cats:
        doc.cats[label] = 1

    return doc

def process_data(data_lines):
    """Processes data lines for text classification."""
    complete_doc_bin = DocBin()
    for line in data_lines:
        data = json.loads(line)
        text = data["text"]
        label = data["label"]
        doc = process_text(text, label)
        complete_doc_bin.add(doc)
    return complete_doc_bin

def main():
    file_path = "output-data/extracted_articles.jsonl"
    data_lines = load_data(file_path, limit=40000)

    training_data, development_data, test_data = split_data(data_lines)

    train_doc_bin = process_data(training_data)
    logging.info("Training data processed and added to DocBin.")
    dev_doc_bin = process_data(development_data)
    logging.info("Development data processed and added to DocBin.")
    test_doc_bin = process_data(test_data)
    logging.info("Test data processed and added to DocBin.")

    train_doc_bin.to_disk("cat-model/prepared-data/train.spacy")
    dev_doc_bin.to_disk("cat-model/prepared-data/dev.spacy")
    test_doc_bin.to_disk("cat-model/prepared-data/test.spacy")
    logging.info("Processed data saved to disk.")

if __name__ == "__main__":
    main()