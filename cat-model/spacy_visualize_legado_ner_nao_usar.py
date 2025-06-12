import json

def pretty_print_jsonl(file_path1, file_path2):
    """
    Prints entities from two JSONL files sequentially for easy comparison, limiting to the first ten items from each file.
    
    Args:
        file_path1 (str): Path to the first JSONL file.
        file_path2 (str): Path to the second JSONL file.
    """
    # Process and print entities from the first file
    print(f"Entities from {file_path1}:")
    print_entities(file_path1)
    
    print("\n" + "=" * 80 + "\n")  # Major separator for clear distinction

    # Process and print entities from the second file
    print(f"Entities from {file_path2}:")
    print_entities(file_path2)

def print_entities(file_path):
    """
    Helper function to print entities from a specified JSONL file, limited to the first ten items.
    
    Args:
        file_path (str): Path to the JSONL file.
    """
    item_count = 0
    max_items = 50  # Maximum number of items to print
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            if item_count >= max_items:
                break  # Stop reading further once we reach the limit
            
            data = json.loads(line)
            text = data['text']
            labels = data['label']
            print(f"Entities from line {item_count} in {file_path}:")
            # Print the entities
            for label in labels:
                start, end, entity_type = label
                entity_text = text[start:end]
                print(f"  {entity_text} [{start}:{end}] - {entity_type}")
            
            print("-" * 80)  # Separator for readability
            item_count += 1  # Increment the counter for each processed line

import json

def print_specific_line_entities(file_path, line_number):
    """
    Prints entities from a specific line in a JSONL file.
    
    Args:
        file_path (str): Path to the JSONL file.
        line_number (int): The line number from which to print entities (1-based index).
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for current_line, line in enumerate(file, start=1):
                if current_line == line_number:
                    data = json.loads(line)
                    text = data['text']
                    labels = data['label']

                    # Print the entities for the specified line
                    print(f"Entities from line {line_number} in {file_path}:")
                    for label in labels:
                        start, end, entity_type = label
                        entity_text = text[start:end]
                        print(f"  {entity_text} [{start}:{end}] - {entity_type}")
                    break
            else:
                print(f"No data found for line {line_number}.")
    except FileNotFoundError:
        print("File not found. Please check the file path and try again.")
    except json.JSONDecodeError:
        print("Error decoding JSON. Please check the file content.")

# Example usage - verificar se arquivos existem
import os

file_path1 = "output-data/results.jsonl"
file_path2 = "output-data/pre_annotated_data.jsonl"

if not os.path.exists(file_path1):
    print(f"Arquivo não encontrado: {file_path1}")
    print("Execute primeiro: python cat-model/spacy_using.py")
else:
    line_number = 95  # Specify the line number you want to print entities from
    print_specific_line_entities(file_path1, line_number)
    
    if os.path.exists(file_path2):
        pretty_print_jsonl(file_path1, file_path2)
    else:
        print(f"Arquivo de comparação não encontrado: {file_path2}")
        print("Exibindo apenas resultados do modelo...")

# Adicionar logs
