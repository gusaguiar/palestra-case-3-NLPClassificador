#!/usr/bin/env python3
"""
Script para verificar a estrutura de pastas e arquivos do projeto.
"""

import os
from pathlib import Path

def check_file_exists(path, description):
    """Verifica se um arquivo existe e exibe status."""
    exists = Path(path).exists()
    status = "✅" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def check_directory_exists(path, description):
    """Verifica se um diretório existe e exibe status."""
    exists = Path(path).is_dir()
    status = "📁" if exists else "❌"
    print(f"{status} {description}: {path}")
    return exists

def main():
    print("🔍 Verificando Estrutura do Projeto")
    print("=" * 50)
    
    # Verificar estrutura de diretórios
    print("\n📂 Estrutura de Diretórios:")
    check_directory_exists("cat-model", "Diretório principal")
    check_directory_exists("cat-model/models", "Diretório de modelos")
    check_directory_exists("cat-model/models/cnn", "Modelo CNN")
    check_directory_exists("cat-model/models/ensemble", "Modelo Ensemble")
    check_directory_exists("cat-model/models/bow", "Modelo BOW")
    check_directory_exists("cat-model/prepared-data", "Dados preparados")
    check_directory_exists("output-data", "Dados de saída")
    check_directory_exists("templates", "Templates web")
    
    # Verificar dados preparados
    print("\n📊 Dados Preparados:")
    train_exists = check_file_exists("cat-model/prepared-data/train.spacy", "Dados de treinamento")
    dev_exists = check_file_exists("cat-model/prepared-data/dev.spacy", "Dados de validação")
    test_exists = check_file_exists("cat-model/prepared-data/test.spacy", "Dados de teste")
    
    # Verificar configuração
    print("\n⚙️ Configuração:")
    config_exists = check_file_exists("cat-model/models/cnn/config.cfg", "Arquivo de configuração")
    
    # Verificar modelos treinados
    print("\n🤖 Modelos Treinados:")
    model_paths = [
        ("cat-model/models/cnn/model-best", "Modelo CNN"),
        ("cat-model/models/ensemble/model-best", "Modelo Ensemble"),
        ("cat-model/models/bow/model-best", "Modelo BOW")
    ]
    
    models_found = []
    for path, description in model_paths:
        if check_directory_exists(path, description):
            models_found.append(path)
    
    # Verificar dados de saída
    print("\n📄 Dados de Saída:")
    check_file_exists("output-data/extracted_articles.jsonl", "Artigos extraídos")
    check_file_exists("output-data/chunkenized_articles.jsonl", "Artigos segmentados")
    check_file_exists("output-data/results.jsonl", "Resultados do modelo")
    
    # Verificar scripts principais
    print("\n🐍 Scripts Principais:")
    check_file_exists("cat-model/spacy_training.py", "Script de treinamento")
    check_file_exists("cat-model/spacy_using.py", "Script de uso")
    check_file_exists("cat-model/spacy_visualize.py", "Script de visualização")
    check_file_exists("web_classifier.py", "Interface web")
    check_file_exists("generate_config.py", "Gerador de configuração")
    
    # Verificar arquivos de configuração
    print("\n📋 Configuração do Projeto:")
    check_file_exists("requirements.txt", "Dependências")
    check_file_exists("README.md", "Documentação")
    check_file_exists("COMO_EXECUTAR_WEB.md", "Instruções web")
    
    # Resumo e próximos passos
    print("\n" + "=" * 50)
    print("📋 RESUMO E PRÓXIMOS PASSOS:")
    print("=" * 50)
    
    if not train_exists or not dev_exists:
        print("❌ Dados não preparados.")
        print("   Execute: python cat-model/spacy_preparation.py")
    
    if not config_exists:
        print("❌ Configuração não encontrada.")
        print("   Execute: python generate_config.py")
    
    if not models_found:
        print("❌ Nenhum modelo treinado encontrado.")
        print("   Execute: python cat-model/spacy_training.py")
    else:
        print(f"✅ {len(models_found)} modelo(s) encontrado(s):")
        for model in models_found:
            print(f"   - {model}")
    
    if train_exists and dev_exists and config_exists:
        print("✅ Pronto para treinamento!")
        print("   Execute: python cat-model/spacy_training.py")
    
    if models_found:
        print("✅ Pronto para usar interface web!")
        print("   Execute: python web_classifier.py")
        print("   Acesse: http://localhost:5000")

if __name__ == "__main__":
    main() 