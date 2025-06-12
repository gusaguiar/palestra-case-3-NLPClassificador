import spacy
from spacy.tokens import DocBin
from spacy.training import Example
from spacy.scorer import Scorer
import logging
import datetime
import os

# Setup logging
# Setup logging
log_directory = 'cat-model/logs/'
current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Criar diretório se não existir
os.makedirs(log_directory, exist_ok=True)

log_filename = f'{log_directory}log_{current_time}.log'
logging.basicConfig(filename=log_filename, level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s')

def load_data_from_spacy_file(file_path, nlp):
    logging.info(f'Loading data from {file_path}')
    doc_bin = DocBin().from_disk(file_path)
    docs = list(doc_bin.get_docs(nlp.vocab))
    logging.info(f'Data loaded successfully with {len(docs)} documents')
    return docs

def evaluate_ner(model_path, test_file_path):
    logging.info(f'Loading model from {model_path}')
    nlp = spacy.load(model_path)
    logging.info('Model loaded successfully')

    logging.info(f'Loading test data from {test_file_path}')
    examples = load_data_from_spacy_file(test_file_path, nlp)
    
    formatted_examples = [Example(predicted=nlp(doc.text), reference=doc) for doc in examples]
    logging.info('Test data formatted into Example objects successfully')

    scorer = Scorer()
    logging.info('Evaluation started')
    scores = scorer.score(formatted_examples)
    logging.info('Evaluation completed successfully')
    logging.info(f'Evaluation scores: {scores}')
    
    return scores['ents_per_type']

# Possíveis caminhos do modelo treinado
possible_model_paths = [
    'cat-model/models/cnn/model-best',
    'cat-model/models/ensemble/model-best',
    'cat-model/models/bow/model-best'
]

# Encontrar modelo existente
model_path = None
for path in possible_model_paths:
    if os.path.exists(path):
        model_path = path
        logging.info(f'Modelo encontrado: {path}')
        break

if model_path is None:
    logging.error('Nenhum modelo treinado encontrado!')
    logging.info('Caminhos verificados:')
    for path in possible_model_paths:
        logging.info(f'  - {path}')
    logging.info('Execute o treinamento primeiro: python cat-model/spacy_training.py')
    exit(1)

# Path to your test dataset stored in .spacy format
test_file_path = 'cat-model/prepared-data/test.spacy'

try:
    entity_scores = evaluate_ner(model_path, test_file_path)
    print("Entity Scores:\n")
    for entity, scores in entity_scores.items():
        print(f"{entity}: Precision={scores['p']:.3f}, Recall={scores['r']:.3f}, F1 Score={scores['f']:.3f}")
except Exception as e:
    logging.error(f'Error during evaluation: {e}')

# Comando alternativo para avaliação:
# python -m spacy evaluate cat-model/models/cnn/model-best/ --output metrics.json cat-model/prepared-data/test.spacy