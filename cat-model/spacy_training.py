import spacy
from pathlib import Path
from spacy.cli.train import train
import spacy_transformers

# Verificar se GPU está disponível
try:
    spacy.require_gpu()
    use_gpu = 0
    print("GPU detectada e será utilizada")
except ValueError:
    print("GPU não disponível, usando CPU")
    use_gpu = -1

# Define paths para estrutura models/cnn
config_path = Path("cat-model/models/cnn/config.cfg")
train_data_path = Path("cat-model/prepared-data/train.spacy")
dev_data_path = Path("cat-model/prepared-data/dev.spacy")
output_path = Path("cat-model/models/cnn")

# Prepare overrides
overrides = {
    "paths.train": str(train_data_path),
    "paths.dev": str(dev_data_path)
}

# Verificar se arquivos existem
if not config_path.exists():
    print(f"Erro: Arquivo de configuração não encontrado: {config_path}")
    print("Execute: python generate_config.py")
    print("Ou manualmente: python -m spacy init config cat-model/models/cnn/config.cfg --lang pt --pipeline textcat")
    exit(1)

if not train_data_path.exists():
    print(f"Erro: Arquivo de treinamento não encontrado: {train_data_path}")
    print("Execute primeiro: python cat-model/spacy_preparation.py")
    exit(1)

if not dev_data_path.exists():
    print(f"Erro: Arquivo de validação não encontrado: {dev_data_path}")
    print("Execute primeiro: python cat-model/spacy_preparation.py")
    exit(1)

# Criar diretório de saída se não existir
output_path.mkdir(exist_ok=True)

print(f"Iniciando treinamento com {'GPU' if use_gpu >= 0 else 'CPU'}")
print(f"Configuração: {config_path}")
print(f"Dados de treino: {train_data_path}")
print(f"Dados de validação: {dev_data_path}")
print(f"Saída: {output_path}")

# Train the model with verbose output
train(config_path, output_path=output_path, overrides=overrides, use_gpu=use_gpu)

print("Training completed! Model saved in:", output_path)
