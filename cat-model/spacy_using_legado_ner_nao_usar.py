import spacy
import json
import logging
from pathlib import Path

def load_model(model_path):
    """
    Load the spaCy model from the given path.
    
    Args:
        model_path (str): Path to the spaCy model directory.
    
    Returns:
        model: Loaded spaCy model.
    """
    spacy.require_gpu()  # Ensure that the GPU resources are utilized   
    logging.info(f"Loading model from {model_path}")
    return spacy.load(model_path)

def process_jsonl_and_save(input_file, output_file, model):
    """
    Process a JSONL file to identify named entities using a spaCy model and save the results.

    Args:
        input_file (str): Path to the input JSONL file.
        output_file (str): Path to save the output JSONL file.
        model: A loaded spaCy NER model.
    """
    logging.info(f"Starting processing of {input_file}")
    # Initialize a counter for reporting
    counter = 0

    with open(input_file, "r", encoding="utf-8") as infile, \
         open(output_file, "w", encoding="utf-8") as outfile:
        
        for line in infile:
            try:
                data = json.loads(line)
                text = data["text"]  # Extract text for NER
                doc = model(text)
                # Extract entities and their positions
                entities = [(ent.text, ent.start_char, ent.end_char, ent.label_) for ent in doc.ents]
                # Update the label with recognized entities
                data["label"] = [[ent[1], ent[2], ent[3]] for ent in entities]
                json.dump(data, outfile)  # Write the modified data with entities to the output file
                outfile.write("\n")
                counter += 1
                if counter % 100 == 0:
                    logging.info(f"Processed {counter} articles")
                    break
            except json.JSONDecodeError as e:
                logging.error(f"Error decoding JSON: {e}")
            except Exception as e:
                logging.error(f"Unexpected error: {e}")

    logging.info(f"Processing complete. Processed data saved to {output_file}")

if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    # Possíveis caminhos do modelo treinado
    possible_model_paths = [
        "cat-model/models/cnn/model-best",
        "cat-model/models/ensemble/model-best",
        "cat-model/models/bow/model-best"
    ]
    
    model_path = None
    for path in possible_model_paths:
        if Path(path).exists():
            model_path = path
            logging.info(f"Modelo encontrado em: {path}")
            break
    
    if model_path is None:
        logging.error("Nenhum modelo treinado encontrado!")
        logging.info("Caminhos verificados:")
        for path in possible_model_paths:
            logging.info(f"  - {path}")
        logging.info("Execute o treinamento primeiro: python cat-model/spacy_training.py")
        exit(1)
    
    nlp_model = load_model(model_path)
    input_path = "output-data/extracted_articles.jsonl"
    output_path = "output-data/results.jsonl"
    process_jsonl_and_save(input_path, output_path, nlp_model)
