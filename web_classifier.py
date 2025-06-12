from flask import Flask, request, jsonify, render_template
import spacy
import json
import logging
import os
from datetime import datetime
import traceback

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)

# Variável global para o modelo
nlp_model = None

# Categorias do modelo
CATEGORIES = [
    "Portaria",
    "Extrato de Contrato", 
    "Extrato de Convênio",
    "Edital",
    "Aviso de Licitação",
    "Resultado de Julgamento",
    "Extrato de Termo Aditivo"
]

def load_classification_model(model_path):
    """
    Carrega o modelo de classificação SpaCy.
    
    Args:
        model_path (str): Caminho para o modelo treinado.
        
    Returns:
        model: Modelo SpaCy carregado.
    """
    try:
        logging.info(f"Carregando modelo de: {model_path}")
        
        # Verificar se o caminho existe
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Modelo não encontrado em: {model_path}")
        
        # Carregar modelo (sem GPU para compatibilidade)
        nlp = spacy.load(model_path)
        
        # Verificar se o modelo tem classificador de texto
        if 'textcat' not in nlp.pipe_names and 'textcat_multilabel' not in nlp.pipe_names:
            logging.warning("Modelo não possui componente de classificação de texto")
        
        logging.info("Modelo carregado com sucesso!")
        return nlp
        
    except Exception as e:
        logging.error(f"Erro ao carregar modelo: {e}")
        raise

def classify_text(text, model):
    """
    Classifica um texto usando o modelo SpaCy.
    
    Args:
        text (str): Texto para classificar.
        model: Modelo SpaCy carregado.
        
    Returns:
        dict: Resultado da classificação com probabilidades.
    """
    try:
        # Processar texto
        doc = model(text)
        
        # Obter scores de classificação
        if hasattr(doc, 'cats') and doc.cats:
            # Ordenar por probabilidade (maior para menor)
            sorted_cats = sorted(doc.cats.items(), key=lambda x: x[1], reverse=True)
            
            # Encontrar categoria com maior probabilidade
            predicted_category = sorted_cats[0][0]
            confidence = sorted_cats[0][1]
            
            return {
                'success': True,
                'predicted_category': predicted_category,
                'confidence': float(confidence),
                'all_probabilities': {cat: float(prob) for cat, prob in sorted_cats},
                'text_length': len(text),
                'processed_at': datetime.now().isoformat()
            }
        else:
            # Fallback se não houver classificação
            return {
                'success': False,
                'error': 'Modelo não possui capacidade de classificação',
                'fallback_category': 'Indefinido',
                'confidence': 0.0,
                'all_probabilities': {cat: 0.0 for cat in CATEGORIES}
            }
            
    except Exception as e:
        logging.error(f"Erro na classificação: {e}")
        return {
            'success': False,
            'error': str(e),
            'fallback_category': 'Erro na classificação',
            'confidence': 0.0
        }

@app.route('/')
def index():
    """Página principal da aplicação."""
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify_endpoint():
    """
    Endpoint para classificar texto.
    
    Espera JSON com campo 'text'.
    Retorna classificação e probabilidades.
    """
    try:
        # Verificar se modelo está carregado
        if nlp_model is None:
            return jsonify({
                'success': False,
                'error': 'Modelo não carregado. Verifique se o caminho do modelo está correto.'
            }), 500
        
        # Obter dados da requisição
        data = request.get_json()
        
        if not data or 'text' not in data:
            return jsonify({
                'success': False,
                'error': 'Campo "text" é obrigatório'
            }), 400
        
        text = data['text'].strip()
        
        if not text:
            return jsonify({
                'success': False,
                'error': 'Texto não pode estar vazio'
            }), 400
        
        if len(text) > 10000:
            return jsonify({
                'success': False,
                'error': 'Texto muito longo (máximo 10.000 caracteres)'
            }), 400
        
        # Classificar texto
        result = classify_text(text, nlp_model)
        
        # Log da classificação
        if result['success']:
            logging.info(f"Texto classificado como: {result['predicted_category']} "
                        f"(confiança: {result['confidence']:.3f})")
        
        return jsonify(result)
        
    except Exception as e:
        logging.error(f"Erro no endpoint de classificação: {e}")
        logging.error(traceback.format_exc())
        return jsonify({
            'success': False,
            'error': 'Erro interno do servidor'
        }), 500

@app.route('/health')
def health_check():
    """Endpoint para verificar saúde da aplicação."""
    model_status = "carregado" if nlp_model is not None else "não carregado"
    
    return jsonify({
        'status': 'ok',
        'model_status': model_status,
        'categories': CATEGORIES,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/model-info')
def model_info():
    """Endpoint com informações do modelo."""
    if nlp_model is None:
        return jsonify({
            'model_loaded': False,
            'error': 'Modelo não carregado'
        })
    
    pipe_names = nlp_model.pipe_names
    
    return jsonify({
        'model_loaded': True,
        'pipe_names': pipe_names,
        'categories': CATEGORIES,
        'has_textcat': 'textcat' in pipe_names or 'textcat_multilabel' in pipe_names,
        'model_lang': nlp_model.lang
    })

def initialize_model():
    """Inicializa o modelo na inicialização da aplicação."""
    global nlp_model
    
    # Possíveis caminhos do modelo (ordem de preferência)
    possible_model_paths = [
        "cat-model/models/cnn/model-best",
        "cat-model/models/ensemble/model-best",
        "cat-model/models/bow/model-best"
    ]
    
    for model_path in possible_model_paths:
        if os.path.exists(model_path):
            try:
                nlp_model = load_classification_model(model_path)
                logging.info(f"Modelo carregado com sucesso de: {model_path}")
                break
            except Exception as e:
                logging.error(f"Falha ao carregar modelo de {model_path}: {e}")
                continue
    
    if nlp_model is None:
        logging.warning("Nenhum modelo foi carregado. A aplicação funcionará em modo de demonstração.")
        logging.info("Caminhos verificados:")
        for path in possible_model_paths:
            logging.info(f"  - {path} {'(existe)' if os.path.exists(path) else '(não existe)'}")

if __name__ == '__main__':
    # Inicializar modelo
    initialize_model()
    
    # Configurações do Flask
    app.config['JSON_AS_ASCII'] = False  # Para suporte a caracteres especiais
    
    logging.info("Iniciando aplicação web...")
    logging.info("Acesse: http://localhost:5002")
    
    # Executar aplicação
    app.run(
        debug=True,
        host='0.0.0.0',
        port=5002,
        use_reloader=False  # Para evitar recarregar o modelo
    ) 