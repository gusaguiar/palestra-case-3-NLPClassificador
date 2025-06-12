#!/usr/bin/env python3
"""
Script para gerar arquivo de configuração do SpaCy para classificação de texto.
"""

import os
from pathlib import Path

def create_config_file():
    """Cria arquivo de configuração básico para classificação de texto."""
    
    config_content = """[system]
gpu_allocator = null
seed = 0

[nlp]
lang = "pt"
pipeline = ["textcat"]
batch_size = 1000
disabled = []
before_creation = null
after_creation = null
after_pipeline_creation = null
tokenizer = {"@tokenizers":"spacy.Tokenizer.v1"}

[components]

[components.textcat]
factory = "textcat"
threshold = 0.5

[components.textcat.model]
@architectures = "spacy.TextCatEnsemble.v2"
nO = null

[components.textcat.model.linear_model]
@architectures = "spacy.TextCatBOW.v2"
exclusive_classes = true
ngram_size = 1
no_output_layer = false

[components.textcat.model.tok2vec]
@architectures = "spacy.Tok2Vec.v2"

[components.textcat.model.tok2vec.embed]
@architectures = "spacy.MultiHashEmbed.v2"
width = 64
attrs = ["NORM","PREFIX","SUFFIX","SHAPE"]
rows = [5000,1000,2500,2500]
include_static_vectors = false

[components.textcat.model.tok2vec.encode]
@architectures = "spacy.MaxoutWindowEncoder.v2"
width = 64
depth = 2
window_size = 1
maxout_pieces = 3

[training]
dev_corpus = "corpora.dev"
train_corpus = "corpora.train"
seed = ${system.seed}
gpu_allocator = ${system.gpu_allocator}
dropout = 0.1
accumulate_gradient = 1
patience = 1600
max_epochs = 0
max_steps = 20000
eval_frequency = 200
frozen_components = []
before_to_disk = null

[training.batcher]
@batchers = "spacy.batch_by_words.v1"
discard_oversize = false
tolerance = 0.2

[training.batcher.size]
@schedules = "compounding.v1"
start = 100
stop = 1000
compound = 1.001

[training.logger]
@loggers = "spacy.ConsoleLogger.v1"
progress_bar = false

[training.optimizer]
@optimizers = "Adam.v1"
beta1 = 0.9
beta2 = 0.999
L2_is_weight_decay = true
L2 = 0.01
grad_clip = 1.0
use_averages = false
eps = 0.00000001
learn_rate = 0.001

[training.score_weights]
cats_score = 1.0
cats_score_desc = null
cats_micro_p = null
cats_micro_r = null
cats_micro_f = null
cats_macro_p = null
cats_macro_r = null
cats_macro_f = null
cats_macro_auc = null
cats_f_per_type = null
cats_auc_per_type = null

[corpora]

[corpora.dev]
@readers = "spacy.Corpus.v1"
path = ${paths.dev}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[corpora.train]
@readers = "spacy.Corpus.v1"
path = ${paths.train}
max_length = 0
gold_preproc = false
limit = 0
augmenter = null

[paths]
train = null
dev = null
vectors = null
init_tok2vec = null
"""

    # Criar diretório se não existir
    config_dir = Path("cat-model/models/cnn")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Escrever arquivo de configuração
    config_path = config_dir / "config.cfg"
    
    with open(config_path, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✅ Arquivo de configuração criado: {config_path}")
    
    # Verificar se dados existem
    train_data = Path("cat-model/prepared-data/train.spacy")
    dev_data = Path("cat-model/prepared-data/dev.spacy")
    
    if train_data.exists() and dev_data.exists():
        print("✅ Dados de treinamento encontrados!")
        print(f"   - Treino: {train_data}")
        print(f"   - Validação: {dev_data}")
        print("\n🚀 Agora você pode executar:")
        print("   python cat-model/spacy_training.py")
    else:
        print("⚠️  Dados de treinamento não encontrados.")
        print("   Execute primeiro: python cat-model/spacy_preparation.py")
    
    return config_path

if __name__ == "__main__":
    print("🔧 Gerando arquivo de configuração do SpaCy...")
    create_config_file() 