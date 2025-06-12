import re
import json
import logging
import multiprocessing
import time
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from bs4 import BeautifulSoup
from collections import defaultdict

# Configure logging com mais detalhes
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - [PID:%(process)d] - %(message)s',
    handlers=[
        logging.FileHandler('chunkenizer_process.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

def initialize_worker():
    """Inicializa worker e configura logging específico."""
    worker_name = multiprocessing.current_process().name
    logging.info(f"Worker {worker_name} iniciado")

def chunkenizer(mp_data):
    """Processa dados e gera chunks com logging detalhado."""
    worker_name = multiprocessing.current_process().name
    start_time = time.time()
    
    processed_data = []
    chunk_size = 400
    chunk_overlap = 50
    
    # Estatísticas para o worker
    total_docs = len(mp_data)
    total_chunks_created = 0
    category_stats = defaultdict(int)
    
    logging.info(f"Worker {worker_name}: Iniciando processamento de {total_docs} documentos")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", "."]
    )
    
    for index, line in enumerate(mp_data):
        try:
            data = json.loads(line)
            text = data["text"]
            labels = data["label"]
            
            # Estatísticas da categoria
            category_stats[labels] += 1
            
            # Split text into chunks
            chunks = text_splitter.split_text(text)
            chunks_count = len(chunks)
            total_chunks_created += chunks_count
            
            chunked_data = []
            for chunk_idx, chunk in enumerate(chunks):
                chunked_data.append({
                    "text": chunk,
                    "label": labels,
                    "original_doc_id": index,
                    "chunk_id": chunk_idx,
                    "total_chunks": chunks_count
                })
            
            for item in chunked_data:
                processed_data.append(json.dumps(item, ensure_ascii=False))
            
            # Log detalhado a cada 1000 documentos
            if (index + 1) % 1000 == 0:
                elapsed_time = time.time() - start_time
                docs_per_second = (index + 1) / elapsed_time if elapsed_time > 0 else 0
                logging.info(
                    f"Worker {worker_name}: Processados {index + 1}/{total_docs} documentos "
                    f"({((index + 1)/total_docs)*100:.1f}%) - "
                    f"Chunks criados: {total_chunks_created} - "
                    f"Velocidade: {docs_per_second:.1f} docs/seg"
                )
        
        except json.JSONDecodeError as e:
            logging.error(f"Worker {worker_name}: Erro JSON na linha {index}: {e}")
            continue
        except Exception as e:
            logging.error(f"Worker {worker_name}: Erro inesperado na linha {index}: {e}")
            continue
    
    # Estatísticas finais do worker
    elapsed_time = time.time() - start_time
    avg_chunks_per_doc = total_chunks_created / total_docs if total_docs > 0 else 0
    
    logging.info(
        f"Worker {worker_name}: Finalizado! "
        f"Docs processados: {total_docs} - "
        f"Chunks criados: {total_chunks_created} - "
        f"Média chunks/doc: {avg_chunks_per_doc:.2f} - "
        f"Tempo: {elapsed_time:.2f}s"
    )
    
    # Log das categorias processadas por este worker
    for category, count in category_stats.items():
        logging.info(f"Worker {worker_name}: Categoria '{category}': {count} documentos")
    
    return processed_data

def process_jsonl_with_multiprocessing(input_file, output_file, num_processes=4):
    """Processa arquivo JSONL com multiprocessing e logging detalhado."""
    start_time = time.time()
    
    logging.info("="*80)
    logging.info("INICIANDO PROCESSAMENTO DE CHUNKENIZAÇÃO")
    logging.info("="*80)
    logging.info(f"Arquivo de entrada: {input_file}")
    logging.info(f"Arquivo de saída: {output_file}")
    logging.info(f"Número de processos: {num_processes}")
    
    # Verificar se arquivo de entrada existe
    try:
        with open(input_file, 'r', encoding='utf-8') as infile:
            lines = infile.readlines()
    except FileNotFoundError:
        logging.error(f"Arquivo não encontrado: {input_file}")
        return
    except Exception as e:
        logging.error(f"Erro ao ler arquivo {input_file}: {e}")
        return
    
    total_lines = len(lines)
    logging.info(f"Total de documentos carregados: {total_lines}")
    
    if total_lines == 0:
        logging.warning("Arquivo de entrada está vazio!")
        return
    
    # Estatísticas iniciais
    category_count = defaultdict(int)
    for line in lines[:min(1000, total_lines)]:  # Amostra para estatísticas
        try:
            data = json.loads(line)
            category_count[data["label"]] += 1
        except:
            continue
    
    logging.info("Distribuição de categorias (amostra):")
    for category, count in category_count.items():
        logging.info(f"  - {category}: {count} documentos")
    
    # Dividir dados entre processos
    mp_data_size = len(lines) // num_processes + (len(lines) % num_processes > 0)
    mp_data = [lines[i:i + mp_data_size] for i in range(0, len(lines), mp_data_size)]
    
    logging.info(f"Dados divididos em {len(mp_data)} grupos:")
    for i, data_chunk in enumerate(mp_data):
        logging.info(f"  - Grupo {i+1}: {len(data_chunk)} documentos")
    
    logging.info("Iniciando processamento multiprocessing...")
    process_start_time = time.time()
    
    # Processar com multiprocessing
    try:
        with multiprocessing.Pool(processes=num_processes, initializer=initialize_worker) as pool:
            results = pool.map(chunkenizer, mp_data)
        
        process_elapsed = time.time() - process_start_time
        logging.info(f"Processamento multiprocessing concluído em {process_elapsed:.2f}s")
        
    except Exception as e:
        logging.error(f"Erro durante processamento multiprocessing: {e}")
        return
    
    # Salvar resultados
    logging.info("Salvando resultados...")
    save_start_time = time.time()
    
    total_processed_lines = 0
    try:
        with open(output_file, 'w', encoding='utf-8') as outfile:
            for result_idx, result in enumerate(results):
                logging.info(f"Salvando resultado do worker {result_idx+1}: {len(result)} chunks")
                for processed_line in result:
                    outfile.write(processed_line + '\n')
                    total_processed_lines += 1
                
                # Log progresso a cada worker salvo
                if (result_idx + 1) % max(1, len(results)//4) == 0:
                    progress = ((result_idx + 1) / len(results)) * 100
                    logging.info(f"Progresso salvamento: {progress:.1f}%")
        
        save_elapsed = time.time() - save_start_time
        logging.info(f"Salvamento concluído em {save_elapsed:.2f}s")
        
    except Exception as e:
        logging.error(f"Erro ao salvar arquivo {output_file}: {e}")
        return
    
    # Estatísticas finais
    total_elapsed = time.time() - start_time
    
    logging.info("="*80)
    logging.info("PROCESSAMENTO CONCLUÍDO COM SUCESSO!")
    logging.info("="*80)
    logging.info(f"Documentos originais: {total_lines}")
    logging.info(f"Chunks gerados: {total_processed_lines}")
    logging.info(f"Fator de expansão: {total_processed_lines/total_lines:.2f}x")
    logging.info(f"Tempo total: {total_elapsed:.2f}s")
    logging.info(f"Velocidade: {total_lines/total_elapsed:.1f} docs/seg")
    logging.info(f"Arquivo salvo em: {output_file}")
    
    # Verificar arquivo de saída
    try:
        file_size = os.path.getsize(output_file)
        logging.info(f"Tamanho do arquivo de saída: {file_size/1024/1024:.2f} MB")
    except:
        pass

def find_data_files():
    """Detecta automaticamente os caminhos corretos dos arquivos."""
    possible_paths = [
        # Executando do diretório raiz
        ('output-data/extracted_articles.jsonl', 'output-data/chunkenized_articles.jsonl'),
        # Executando do diretório cat-model
        ('../output-data/extracted_articles.jsonl', '../output-data/chunkenized_articles.jsonl'),
        # Caminhos absolutos como fallback
        ('../../output-data/extracted_articles.jsonl', '../../output-data/chunkenized_articles.jsonl')
    ]
    
    for input_path, output_path in possible_paths:
        if os.path.exists(input_path):
            logging.info(f"Arquivos encontrados: {input_path}")
            return input_path, output_path
    
    return None, None

if __name__ == '__main__':
    logging.info("Iniciando script spacy_chunkenizer.py")
    logging.info(f"PID do processo principal: {multiprocessing.current_process().pid}")
    logging.info(f"Diretório atual: {os.getcwd()}")
    
    # Detectar caminhos automaticamente
    input_path, output_path = find_data_files()
    
    if input_path is None:
        logging.error("Arquivo extracted_articles.jsonl não encontrado!")
        logging.error("Caminhos verificados:")
        logging.error("  - output-data/extracted_articles.jsonl")
        logging.error("  - ../output-data/extracted_articles.jsonl")
        logging.error("  - ../../output-data/extracted_articles.jsonl")
        logging.error("Verifique se está executando do diretório correto ou se o arquivo existe")
        exit(1)
    
    logging.info(f"Caminho detectado para entrada: {input_path}")
    logging.info(f"Caminho detectado para saída: {output_path}")
    
    process_jsonl_with_multiprocessing(input_path, output_path, num_processes=16)
    
    logging.info("Script finalizado!")
