from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import os
from pathlib import Path
from predictor import (
    predict, load_model, word2idx, idx2tag, num_tags,
    predict_phobert, load_phobert_model, tokenize
)

app = Flask(__name__)
CORS(app)

base_dir = Path(__file__).parent.absolute()
bilstm_model_path = base_dir / "artifacts" / "bilstm" / "bilstm_crf.pt"
phobert_model_dir = base_dir / "artifacts" / "phobert"

# ============ Model Loading ============
BILSTM_MODEL_LOADED = False
PHOBERT_MODEL_LOADED = False
bilstm_model = None
phobert_model = None
phobert_tokenizer = None

# Load BiLSTM-CRF
try:
    bilstm_model = load_model(
        str(bilstm_model_path),
        vocab_size=len(word2idx),
        num_tags=num_tags
    )
    print("✓ BiLSTM-CRF model loaded successfully")
    BILSTM_MODEL_LOADED = True
except Exception as e:
    print(f"✗ Error loading BiLSTM-CRF: {e}")
    BILSTM_MODEL_LOADED = False

# Load PhoBERT (lazy loading)
try:
    phobert_model, phobert_tokenizer = load_phobert_model()
    print("✓ PhoBERT model loaded successfully")
    PHOBERT_MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  PhoBERT not available: {e}")
    PHOBERT_MODEL_LOADED = False

# ============ Endpoints ============

@app.route("/", methods=["GET"])
def home():
    """Health check endpoint"""
    return jsonify({
        "status": "running",
        "models": {
            "bilstm_crf": BILSTM_MODEL_LOADED,
            "phobert": PHOBERT_MODEL_LOADED
        },
        "version": "2.0"
    })

@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict NER labels - POST: {"text": "...", "model": "phobert|bilstm"}"""
    try:
        data = request.get_json()
        if not data or "text" not in data:
            return jsonify({"success": False, "error": "Missing 'text' field"}), 400
        
        text = data.get("text", "").strip()
        model_choice = data.get("model", "phobert").lower()
        
        if not text:
            return jsonify({"success": False, "error": "Text cannot be empty"}), 400
        if len(text) > 5000:
            return jsonify({"success": False, "error": "Text too long (max 5000 chars)"}), 400
        
        if model_choice == "phobert":
            if not PHOBERT_MODEL_LOADED:
                return jsonify({"success": False, "error": "PhoBERT not loaded"}), 503
            result = predict_phobert(text, phobert_model, phobert_tokenizer)
        elif model_choice in ["bilstm", "bilstm_crf"]:
            if not BILSTM_MODEL_LOADED or bilstm_model is None:
                return jsonify({"success": False, "error": "BiLSTM-CRF not loaded"}), 503
            result = predict(text, bilstm_model, model_name="BiLSTM-CRF")
        else:
            return jsonify({"success": False, "error": f"Unknown model: {model_choice}"}), 400
        
        return jsonify({"success": True, "data": result})
    except Exception as e:
        print(f"Error in /api/predict: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"success": False, "error": "Endpoint not found"}), 404

@app.errorhandler(500)
def server_error(error):
    print(f"Server error: {error}")
    return jsonify({"success": False, "error": "Internal server error"}), 500

if __name__ == "__main__":
    print("🚀 NER Backend API v2.0")
    print(f"BiLSTM-CRF loaded: {BILSTM_MODEL_LOADED}")
    print(f"PhoBERT loaded: {PHOBERT_MODEL_LOADED}")
    if BILSTM_MODEL_LOADED:
        print(f"  - Vocabulary size: {len(word2idx)}")
        print(f"  - Number of tags: {num_tags}")
        print(f"  - Available labels: {list(idx2tag.values())[:5]}...")
    print()
    debug_mode = os.getenv("FLASK_ENV", "production") == "development"
    app.run(host="0.0.0.0", port=5000, debug=debug_mode, use_reloader=debug_mode)
