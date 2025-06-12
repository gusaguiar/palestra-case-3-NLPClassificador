import os
import logging
from xml.etree import ElementTree as ET
from collections import defaultdict
import json
from bs4 import BeautifulSoup
import re
import multiprocessing

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logging.info("Logging is configured correctly.")

TARGET_ART_TYPES = {
    "Extrato de Contrato",
    "Extrato de Termo Aditivo",
    "Aviso de Licitação",
    "Portaria",
    "Edital",
    "Resultado de Julgamento",
    "Extrato de Convênio"
}

def clean_text(text):
    """Removes HTML tags and normalizes whitespace."""
    soup = BeautifulSoup(text, "html.parser")
    cleaned_text = soup.get_text(" ")
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text).strip()
    return cleaned_text

def extract_article_details(xml_path):
    """Extracts the article details from an XML file."""
    if not os.path.exists(xml_path):
        logging.error("File does not exist: %s", xml_path)
        return None

    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        article = root.find('.//article')
        artType = article.get('artType') if article else None
        if artType in TARGET_ART_TYPES:
            text_element = root.find('.//Texto')
            text_content = text_element.text if text_element is not None else ""
            article_details = {
                'label': artType,
                'text': clean_text(text_content)
            }
            logging.debug("Extracted artType: %s from file: %s", artType, xml_path)
            return article_details
        else:
            return None
    except ET.ParseError as e:
        logging.error("XML parsing error at %s: %s", xml_path, e)
        return None
    except Exception as e:
        logging.error("Error processing %s: %s", xml_path, e)
        return None

def process_directory_parallel(root_directory):
    """Processes XML files from a directory using multiprocessing."""
    logging.info("Process started for directory: %s", root_directory)
    if not os.path.exists(root_directory):
        logging.error("Root directory not found: %s", root_directory)
        return

    xml_paths = []
    for dirpath, _, filenames in os.walk(root_directory):
        for filename in filenames:
            if filename.endswith('.xml'):
                xml_paths.append(os.path.join(dirpath, filename))

    logging.info("Total XML files found: %d", len(xml_paths))
    xml_paths = xml_paths[:5000]  # Processa apenas 5000 arquivos para teste
    logging.info("Limited to %d files for testing", len(xml_paths))

    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        articles = pool.map(extract_article_details, xml_paths)

    articles = [article for article in articles if article is not None]

    artType_collections = defaultdict(list)
    for article in articles:
        if len(artType_collections[article['label']]) < 1000:
            artType_collections[article['label']].append(article)

    for artType, collected_articles in artType_collections.items():
        logging.info("Collected %d articles for artType: %s", len(collected_articles), artType)

    return artType_collections

def save_collections_to_jsonl(artType_collections, output_file_path):
    """Saves the collected articles to a single JSONL file."""
    with open(output_file_path, 'w', encoding='utf-8') as f:
        for articles in artType_collections.values():
            for article in articles:
                json_line = json.dumps(article, ensure_ascii=False)
                f.write(json_line + '\n')
    logging.info("Saved collected articles to %s", output_file_path)

if __name__ == '__main__':
    try:
        root_directory = os.path.abspath('data')
        output_directory = os.path.abspath('Categoria/output-data')
        output_file_path = os.path.join(output_directory, "extracted_articles.jsonl")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)
            logging.info("Output directory created: %s", output_directory)

        artType_collections = process_directory_parallel(root_directory)
        save_collections_to_jsonl(artType_collections, output_file_path)
    except Exception as e:
        logging.error("An error occurred: %s", e)
        raise
