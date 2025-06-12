# Classificador Automático de Documentos Oficiais

## O que é este projeto?

Este projeto é um sistema inteligente que lê documentos oficiais em português e automaticamente identifica que tipo de documento é. É como ter um assistente que consegue distinguir entre diferentes tipos de documentos do governo sem precisar ler o conteúdo completo.

## Para que serve?

O sistema consegue classificar automaticamente 7 tipos diferentes de documentos oficiais:
- **Portaria** - Regras e normas do governo
- **Extrato de Contrato** - Resumos de contratos com empresas
- **Extrato de Convênio** - Resumos de parcerias entre órgãos
- **Edital** - Chamadas para participar de licitações
- **Aviso de Licitação** - Avisos sobre processos de compra do governo
- **Resultado de Julgamento** - Resultados de quem ganhou as licitações
- **Extrato de Termo Aditivo** - Mudanças feitas em contratos existentes

## Como funciona?

O sistema usa inteligência artificial para analisar o texto e identificar padrões que são característicos de cada tipo de documento. É como ensinar um computador a reconhecer diferentes tipos de texto da mesma forma que uma pessoa experiente faria.

## Organização dos Arquivos

```
palestra-case-3-NLPClassificador/
├── cat-model/                          # Pasta principal com os modelos
│   ├── models/                          # Modelos de inteligência artificial treinados
│   │   ├── cnn/                         # Modelo CNN (mais eficiente)
│   │   ├── ensemble/                    # Modelo ensemble (combina vários)
│   │   └── bow/                         # Modelo Bag-of-Words (mais simples)
│   ├── prepared-data/                   # Dados prontos para treinar
│   │   ├── train.spacy                  # Dados para treinar (11MB)
│   │   ├── dev.spacy                    # Dados para validar (1.4MB)
│   │   └── test.spacy                   # Dados para testar (1.4MB)
│   ├── logs/                            # Arquivos de registro do que aconteceu
│   ├── import_data_cat.py               # Extrai dados dos arquivos XML
│   ├── spacy_preparation.py             # Prepara os dados para treinar
│   ├── spacy_training.py                # Treina o modelo
│   ├── spacy_evaluation.py              # Testa qual é a precisão do modelo
│   └── aux_clean_memory.py              # Limpa a memória do computador
├── templates/                           # Arquivos da interface web
│   └── index.html                       # Página web do sistema
├── output-data/                         # Dados que o sistema produz
│   └── extracted_articles.jsonl         # Documentos extraídos (12MB)
├── web_classifier.py                    # Servidor web para usar o sistema
├── generate_config.py                   # Gera configurações automaticamente
├── requirements.txt                     # Lista de programas necessários
├── metrics.json                         # Resultados dos testes de precisão
└── README.md                            # Este arquivo explicativo
```

## Como usar o sistema?

### O que você precisa ter instalado

Antes de usar o sistema, certifique-se de ter:

```powershell
# Python 3.8 ou mais recente
# Instalar as bibliotecas necessárias
pip install spacy spacy-transformers
pip install beautifulsoup4 langchain flask
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Baixar o modelo de português do SpaCy
python -m spacy download pt_core_news_lg
```

## Forma mais fácil de usar: Interface Web

A maneira mais simples de usar o sistema é através da interface web:

```powershell
python web_classifier.py
```

Depois abra seu navegador e acesse: http://localhost:5000

Você poderá colar qualquer texto de documento oficial e o sistema dirá que tipo de documento é.

## Para Desenvolvedores: Como treinar seu próprio modelo

Se você quiser treinar o sistema com seus próprios documentos, siga estes passos:

### Passo 1: Extrair dados dos arquivos

Este comando lê arquivos XML e extrai o texto dos documentos:

```powershell
cd cat-model
python import_data_cat.py
```

O que este comando faz:
- Lê arquivos XML com documentos oficiais
- Extrai apenas o texto, removendo códigos HTML
- Organiza os documentos por categoria
- Salva tudo em um arquivo mais fácil de processar

### Passo 2: Preparar os dados para treinar

Este comando organiza os dados de uma forma que o computador entende melhor:

```powershell
python spacy_preparation.py
```

IMPORTANTE: Este sistema trabalha com documentos completos, não fragmenta os textos em pedaços menores.

O que este comando faz:
- Converte textos para minúsculas (para padronizar)
- Remove elementos desnecessários como emails e URLs
- Remove palavras comuns que não ajudam na classificação
- Separa os documentos em 3 grupos: 80% para treinar, 10% para validar, 10% para testar

### Passo 3: Treinar o modelo

Este comando ensina o computador a reconhecer os diferentes tipos de documento:

```powershell
python spacy_training.py
```

O que acontece durante o treinamento:
- O computador analisa milhares de documentos de exemplo
- Aprende quais palavras e padrões são típicos de cada categoria
- Testa seu aprendizado constantemente para melhorar
- Usa a placa de vídeo (GPU) se disponível para ser mais rápido

### Passo 4: Testar a precisão

Este comando verifica se o modelo está funcionando bem:

```powershell
python spacy_evaluation.py
```

O que este teste faz:
- Usa documentos que o modelo nunca viu antes
- Verifica se o modelo consegue identificar corretamente cada tipo
- Calcula porcentagens de acerto para cada categoria
- Mostra um relatório detalhado com os resultados

## Resumo: Como usar este sistema

Existem duas formas principais de usar este projeto:

### Forma Simples (Recomendada)
Se você só quer classificar documentos:
1. Execute: `python web_classifier.py`
2. Abra http://localhost:5000 no navegador
3. Cole o texto do documento
4. Veja o resultado instantaneamente

### Forma Avançada (Para Desenvolvedores)
Se você quer treinar o modelo com seus próprios dados:
1. **Extrair dados**: `python cat-model/import_data_cat.py`
2. **Preparar dados**: `python cat-model/spacy_preparation.py`
3. **Treinar modelo**: `python cat-model/spacy_training.py`
4. **Testar precisão**: `python cat-model/spacy_evaluation.py`
5. **Usar interface**: `python web_classifier.py`

IMPORTANTE: Para classificar documentos, você só precisa seguir a "Forma Simples". Os arquivos removidos (spacy_using.py e spacy_visualize.py) eram para uma funcionalidade diferente (identificar nomes, CPFs, etc.) e não são necessários para classificar tipos de documento.

## Detalhes da Interface Web

### Como funciona a interface?

A interface web é uma página simples onde você pode testar o sistema:

1. **Cole ou digite** o texto de qualquer documento oficial
2. **Clique em "Classificar"** ou use Ctrl+Enter
3. **Veja o resultado**: O sistema mostra qual tipo de documento é e com que porcentagem de certeza

### Exemplo prático

Se você colar este texto:
```
PORTARIA Nº 123, DE 15 DE JUNHO DE 2024

O SECRETÁRIO DE ESTADO DA FAZENDA, no uso das atribuições que lhe confere...
```

O sistema responderá algo como:
- **Resultado**: Portaria (97,5% de certeza)
- **Outras possibilidades**: Edital (1,2%), Extrato de Contrato (0,8%)

### Atalhos úteis

- **Ctrl+Enter**: Classifica o texto rapidamente
- **Botão Limpar**: Remove o texto e resultados
- **Status colorido**: Verde = funcionando, Amarelo = modo demonstração, Vermelho = problema

### Para programadores: API

Se você quiser integrar este sistema com outros programas, pode usar estas URLs:

- `GET /` - Página principal
- `POST /classify` - Enviar texto para classificar
- `GET /health` - Verificar se está funcionando
- `GET /model-info` - Obter informações técnicas

## Quão preciso é o sistema?

O sistema atual é extremamente preciso. Ele foi testado com milhares de documentos que nunca tinha visto antes e conseguiu classificar corretamente quase 100% dos casos.

### Precisão por tipo de documento

| Tipo de Documento | Acertos | 
|-------------------|---------|
| Portaria | 100% |
| Edital | 100% |
| Extrato de Termo Aditivo | 100% |
| Extrato de Contrato | 99,5% |
| Aviso de Licitação | 99,5% |
| Extrato de Convênio | 99,4% |
| Resultado de Julgamento | 99,5% |

### O que significam estes números?

- **99,7% de precisão geral**: De cada 1000 documentos, o sistema erra apenas 3
- **Velocidade**: Processa mais de 35.000 palavras por segundo
- **Confiabilidade**: Funciona consistentemente bem com documentos novos

Estes resultados mostram que o sistema está pronto para uso real em ambientes profissionais.

## Informações Técnicas

### Dados utilizados para treinar

O sistema foi treinado com uma grande quantidade de documentos reais:

- **Total de documentos**: 36.362 documentos oficiais
- **Para treinamento**: 29.089 documentos (80%) - para ensinar o sistema
- **Para validação**: 3.636 documentos (10%) - para ajustar durante o treinamento  
- **Para teste**: 3.637 documentos (10%) - para testar a precisão final

### Limpeza de memória

Se o sistema estiver lento ou com problemas de memória, execute:
```powershell
python cat-model/aux_clean_memory.py
```
Este comando limpa a memória e pode resolver problemas de performance.

## Conclusão

Este é um sistema completo e funcional para classificação automática de documentos oficiais. Com uma precisão de 99,7%, ele está pronto para ser usado em ambiente real para automatizar a classificação de grandes volumes de documentos governamentais.

O sistema é especialmente útil para:
- Órgãos públicos que precisam organizar grandes volumes de documentos
- Empresas que trabalham com licitações e contratos públicos
- Pesquisadores que estudam documentos governamentais
- Qualquer pessoa que precisa identificar rapidamente tipos de documentos oficiais 