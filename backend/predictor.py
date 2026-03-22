import pickle
import torch
import torch.nn as nn
from torchcrf import CRF
import vncorenlp
import os
import json
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForTokenClassification

# ============ BiLSTM-CRF Artifacts ============
def load_artifacts():
    """Load BiLSTM word and label mappings"""
    base_dir = Path(__file__).parent.absolute()
    word2idx_path = base_dir / "artifacts" / "bilstm" / "word2idx.pkl"
    if not word2idx_path.exists():
        raise FileNotFoundError(f"Missing {word2idx_path}")
    with open(word2idx_path, "rb") as f:
        word2idx = pickle.load(f)
    idx2tag_path = base_dir / "artifacts" / "bilstm" / "idx2tag.pkl"
    if not idx2tag_path.exists():
        raise FileNotFoundError(f"Missing {idx2tag_path}")
    with open(idx2tag_path, "rb") as f:
        idx2tag = pickle.load(f)
    return word2idx, idx2tag

word2idx, idx2tag = load_artifacts()
tag2idx = {v: k for k, v in idx2tag.items()}
num_tags = len(idx2tag)

# ============ PhoBERT Tag Vocab ============
def load_phobert_tag_vocab():
    """Load PhoBERT tag vocab from config.json"""
    base_dir = Path(__file__).parent.absolute()
    config_path = base_dir / "artifacts" / "phobert" / "config.json"
    
    if not config_path.exists():
        raise FileNotFoundError(f"Missing {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    id2label = {}
    if "id2label" in config:
        id2label = {int(k): v for k, v in config["id2label"].items()}
    
    if not id2label:
        raise ValueError("No id2label mapping in PhoBERT config!")
    
    label2id = config.get("label2id", {})
    return id2label, label2id

try:
    phobert_id2label, phobert_label2id = load_phobert_tag_vocab()
    print(f"✓ PhoBERT labels loaded: {len(phobert_id2label)} tags")
except Exception as e:
    print(f"✗ Error loading PhoBERT vocab: {e}")
    raise

def init_vncorenlp():
    """Initialize VnCoreNLP for Vietnamese word segmentation"""
    try:
        jar_path = os.path.join(os.path.dirname(__file__), "vncorenlp", "VnCoreNLP-1.1.1.jar")
        if not os.path.exists(jar_path):
            return None
        return vncorenlp.VnCoreNLP(jar_path, annotators="wseg", max_heap_size='-Xmx500m')
    except:
        return None

rdrsegmenter = init_vncorenlp()

def tokenize(sentence: str):
    """Vietnamese word segmentation using VnCoreNLP"""
    if rdrsegmenter is not None:
        try:
            result = rdrsegmenter.tokenize(sentence)
            tokens = [tok for sent in result for tok in sent]
            return tokens
        except:
            pass
    return sentence.split()

def sentence_to_tensor(tokens):
    """Convert tokens to tensor indices"""
    UNK_IDX = word2idx.get("<UNK>", 1)
    indices = [word2idx.get(t, UNK_IDX) for t in tokens]
    return torch.tensor([indices], dtype=torch.long)

class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, pretrained_weights=None, hidden_size=128, embedding_dim=300):
        super(BiLSTM_CRF, self).__init__()
        
        # Embedding layer
        if pretrained_weights is not None:
            self.embedding = nn.Embedding.from_pretrained(
                embeddings=pretrained_weights,
                freeze=False,
                padding_idx=0
            )
        else:
            self.embedding = nn.Embedding(
                vocab_size, 
                embedding_dim,
                padding_idx=0
            )
        
        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            bidirectional=True,
            batch_first=True
        )
        
        # Linear layer
        self.hidden2tag = nn.Linear(hidden_size * 2, tagset_size)
        
        # CRF layer
        self.crf = CRF(tagset_size, batch_first=True)
    
    def forward(self, sentence, tags=None, mask=None):
        """Args: sentence [batch_size, seq_len], tags [batch_size, seq_len] (optional), mask [batch_size, seq_len]"""
        embeds = self.embedding(sentence)
        lstm_out, _ = self.lstm(embeds)
        linear_out = self.hidden2tag(lstm_out)
        if tags is not None:
            loss = -self.crf(linear_out, tags, mask=mask, reduction='mean')
            return loss
        else:
            return self.crf.decode(linear_out, mask=mask)

def load_model(model_path, vocab_size, num_tags):
    """Load BiLSTM-CRF model"""
    model = BiLSTM_CRF(vocab_size, num_tags, hidden_size=128)
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint)
        print(f"✓ Model loaded: {model_path}")
    model.eval()
    return model

def predict(sentence: str, model, model_name: str = "BiLSTM-CRF"):
    """Named Entity Recognition prediction"""
    tokens = tokenize(sentence)
    if not tokens:
        return {
            "model": model_name,
            "text": sentence,
            "tokens": [],
            "labels": [],
            "entities": [],
            "confidence_scores": []
        }
    x = sentence_to_tensor(tokens)
    with torch.no_grad():
        predictions = model(x)
    tag_indices = predictions[0]
    labels = [idx2tag.get(i, "O") for i in tag_indices]
    # BiLSTM uses CRF which optimizes for valid sequences, so confidence is uniform (1.0)
    confidence_scores = [1.0] * len(labels)
    entities = extract_entities(tokens, labels, sentence, confidences=confidence_scores)
    return {
        "model": model_name,
        "text": sentence,
        "tokens": tokens,
        "labels": labels,
        "entities": entities,
        "confidence_scores": confidence_scores
    }

def normalize_label(label):
    """Fix typo PROUCT → PRODUCT and handle prefixes"""
    if label is None:
        return None
    label = label.replace("PROUCT", "PRODUCT")
    if label.startswith("B-") or label.startswith("I-"):
        return label[:2] + label[2:].replace("PROUCT", "PRODUCT")
    return label

# ==================== PRICE PATTERN MATCHING ====================
import re

def is_price_token(token):
    """Check if token is currency unit"""
    return token.lower() in {'triệu', 'k', 'đồng', 'vnd', 'vnđ', 'tr', 'đ', 'ngàn', 'nghìn', 'tỷ', 'tỉ', 'vn', 'tỏi', 'củ', 'cành', 'lít'}

def is_numeric_token(token):
    """Check if token contains numbers"""
    return bool(re.search(r'\d+', token))

def extract_entities(tokens, labels, original_text, confidences=None):
    """Parse BIO tags to entities with PRICE pattern matching and confidence scores"""
    if confidences is None:
        confidences = [1.0] * len(labels)
    
    corrected_labels = list(labels)
    shorthand_conversions = {}
    for i, token in enumerate(tokens):
        match_single = re.match(r'^(\d+[.,]?\d*)\s*([kmđ]|triệu|tr)$', token, re.IGNORECASE)
        if match_single:
            corrected_labels[i] = "B-PRICE"
            continue
        match_shorthand = re.match(r'^(\d+)\s*([mtrđk]|tr)\s*(\d+)$', token, re.IGNORECASE)
        if match_shorthand:
            main, unit, decimal = match_shorthand.groups()
            corrected_labels[i] = "B-PRICE"
            # Keep original format: 2m6 stays 2m6
            shorthand_conversions[i] = token
            if i + 1 < len(tokens) and is_price_token(tokens[i + 1]):
                corrected_labels[i + 1] = "O"
            continue
    
    entities = []
    current_entity = None
    current_label = None
    current_tokens = []
    current_confidences = []
    current_token_idx = None
    
    for token_idx, (token, label, conf) in enumerate(zip(tokens, corrected_labels, confidences)):
        if label.startswith("B-"):
            if current_entity is not None:
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0.0
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": normalize_label(current_label),
                    "tokens": current_tokens,
                    "token_indices": list(range(current_token_idx, token_idx)),
                    "confidence": avg_conf
                })
            
            current_label = label[2:]
            current_tokens = [token]
            current_confidences = [conf]
            current_entity = token_idx
            current_token_idx = token_idx
        
        elif label.startswith("I-"):
            entity_label = label[2:]
            
            if current_entity is None:
                current_label = entity_label
                current_tokens = [token]
                current_confidences = [conf]
                current_entity = token_idx
                current_token_idx = token_idx
            elif entity_label == current_label:
                current_tokens.append(token)
                current_confidences.append(conf)
            else:
                if current_entity is not None:
                    avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0.0
                    entities.append({
                        "text": " ".join(current_tokens),
                        "label": normalize_label(current_label),
                        "tokens": current_tokens,
                        "token_indices": list(range(current_token_idx, token_idx)),
                        "confidence": avg_conf
                    })
                
                current_label = entity_label
                current_tokens = [token]
                current_confidences = [conf]
                current_entity = token_idx
                current_token_idx = token_idx
        
        else:  # O-tag
            if current_entity is not None:
                avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0.0
                entities.append({
                    "text": " ".join(current_tokens),
                    "label": normalize_label(current_label),
                    "tokens": current_tokens,
                    "token_indices": list(range(current_token_idx, token_idx)),
                    "confidence": avg_conf
                })
            
            current_entity = None
            current_label = None
            current_tokens = []
            current_confidences = []
            current_token_idx = None
    
    if current_entity is not None:
        avg_conf = sum(current_confidences) / len(current_confidences) if current_confidences else 0.0
        entities.append({
            "text": " ".join(current_tokens),
            "label": normalize_label(current_label),
            "tokens": current_tokens,
            "token_indices": list(range(current_token_idx, len(tokens))),
            "confidence": avg_conf
        })
    
    # Apply shorthand conversions
    for ent in entities:
        indices = ent.get('token_indices', [])
        if ent['label'] == 'PRICE' and len(ent['tokens']) == 1 and len(indices) > 0:
            token_idx = indices[0]
            if token_idx in shorthand_conversions:
                ent['text'] = shorthand_conversions[token_idx]
    
    # PRICE validation & cleanup
    valid_units = {'triệu', 'k', 'đồng', 'vnd', 'vnđ', 'tr', 'đ', 'ngàn', 'nghìn', 'tỷ', 'tỉ', 'vn', 'tỏi', 'củ', 'cành', 'lít'}
    filtered_entities = []
    used_indices = set()
    
    for ent in entities:
        indices = ent.get('token_indices', [])
        if ent['label'] == 'PRICE':
            if ent['tokens']:
                last_token = ent['tokens'][-1].lower()
                first_token = ent['tokens'][0]
                
                is_valid = False
                if len(ent['tokens']) >= 2:
                    if last_token in valid_units and is_numeric_token(first_token):
                        is_valid = True
                elif len(ent['tokens']) == 1:
                    token = ent['tokens'][0]
                    match_single = re.match(r'^(\d+[.,]?\d*)\s*([kmđ]|triệu|tr)$', token, re.IGNORECASE)
                    match_shorthand = re.match(r'^(\d+)\s*([mtrđk]|tr)\s*(\d+)$', token, re.IGNORECASE)
                    if match_single or match_shorthand:
                        is_valid = True
                
                for token in ent['tokens']:
                    if '_' in token and token.lower() not in valid_units and token.lower() not in {'hà_nội', 'hcm', 'tphcm'}:
                        is_valid = False
                        break
                
                if is_valid:
                    filtered_entities.append(ent)
                    used_indices.update(indices)
            else:
                filtered_entities.append(ent)
                used_indices.update(indices)
        else:
            filtered_entities.append(ent)
            used_indices.update(indices)
    
    # Add missed PRICE patterns
    for i in range(len(tokens) - 1):
        if i in used_indices or (i + 1) in used_indices:
            continue
        
        curr_token = tokens[i]
        next_token = tokens[i + 1]
        # Detect three-token patterns like: <number> <unit> <number>
        # e.g. "10 tỏi 5", "10 củ 4" -> treat as single PRICE entity
        if (i + 2) < len(tokens) and i not in used_indices and (i + 1) not in used_indices and (i + 2) not in used_indices:
            third_token = tokens[i + 2]
            if is_numeric_token(curr_token) and is_price_token(next_token) and is_numeric_token(third_token):
                if '_' not in next_token or next_token.lower() in valid_units:
                    filtered_entities.append({
                        "text": f"{curr_token} {next_token} {third_token}",
                        "label": "PRICE",
                        "tokens": [curr_token, next_token, third_token],
                        "token_indices": [i, i+1, i+2]
                    })
                    used_indices.update([i, i+1, i+2])
                    continue
        
        if is_numeric_token(curr_token) and is_price_token(next_token):
            if '_' not in next_token or next_token.lower() in valid_units:
                filtered_entities.append({
                    "text": f"{curr_token} {next_token}",
                    "label": "PRICE",
                    "tokens": [curr_token, next_token],
                    "token_indices": [i, i+1]
                })
                used_indices.update([i, i+1])
        
        elif re.match(r'^\d+[.,]\d+$', curr_token) and is_price_token(next_token):
            if '_' not in next_token or next_token.lower() in valid_units:
                filtered_entities.append({
                    "text": f"{curr_token} {next_token}",
                    "label": "PRICE",
                    "tokens": [curr_token, next_token],
                    "token_indices": [i, i+1]
                })
                used_indices.update([i, i+1])
    
    # Add single-token PRICE patterns
    for i, token in enumerate(tokens):
        if i in used_indices:
            continue
        
        match_single = re.match(r'^(\d+[.,]?\d*)\s*([kmđ]|triệu|tr|ngàn|vnđ|củ|cành|lít|nghìn|vnd)$', token, re.IGNORECASE)
        if match_single:
            num, unit = match_single.groups()
            if unit.lower() in {'k', 'm', 'đ', 'triệu', 'tr', 'ngàn', 'nghìn', 
                                'vnd', 'vnđ', 'đồng', 'tỷ', 'tỉ', 'củ', 'tỏi', 'lít'}:
                unit_display = {'k': 'k', 'ngàn': 'ngàn', 'nghìn': 'nghìn',
                                'm': 'm', 'triệu': 'triệu', 'tr': 'tr', 'củ': 'củ',
                                'đ': 'đ', 'đồng': 'đồng', 'vnd': 'vnd', 'vnđ': 'vnđ',
                                'tỷ': 'tỷ', 'tỉ': 'tỉ', 'tỏi': 'tỏi',
                                'lít': 'lít'}.get(unit.lower(), unit)
                filtered_entities.append({
                    "text": f"{num} {unit_display}",
                    "label": "PRICE",
                    "tokens": [token],
                    "token_indices": [i]
                })
                used_indices.add(i)
        
        match_shorthand = re.match(r'^(\d+)\s*([mtrđk]|tr)\s*(\d+)$', token, re.IGNORECASE)
        if match_shorthand:
            # Keep original format: 2m6 stays 2m6
            filtered_entities.append({
                "text": token,
                "label": "PRICE",
                "tokens": [token],
                "token_indices": [i]
            })
            used_indices.add(i)
    
    filtered_entities.sort(key=lambda e: min(e.get('token_indices', [float('inf')])))
    
    for ent in filtered_entities:
        ent.pop('token_indices', None)
    
    return filtered_entities

# ============ PhoBERT Model Loading ============
_phobert_model = None
_phobert_tokenizer = None

def load_phobert_model():
    """Load PhoBERT model from artifacts/phobert"""
    global _phobert_model, _phobert_tokenizer
    
    if _phobert_model is not None:
        return _phobert_model, _phobert_tokenizer
    
    base_dir = Path(__file__).parent.absolute()
    model_dir = base_dir / "artifacts" / "phobert"
    
    if not model_dir.exists():
        raise FileNotFoundError(f"PhoBERT model not found: {model_dir}")
    
    print(f"Loading PhoBERT from {model_dir}...")
    _phobert_tokenizer = AutoTokenizer.from_pretrained(str(model_dir))
    _phobert_model = AutoModelForTokenClassification.from_pretrained(str(model_dir))
    _phobert_model.eval()
    print("✓ PhoBERT model loaded successfully")
    return _phobert_model, _phobert_tokenizer

def predict_phobert(text: str, model=None, tokenizer=None, debug=False):
    """PhoBERT Named Entity Recognition with confidence tracking"""
    if model is None or tokenizer is None:
        model, tokenizer = load_phobert_model()
    
    tokens = tokenize(text)
    
    try:
        encodings = tokenizer(
            tokens,
            truncation=True,
            is_split_into_words=True,
            max_length=512,
            return_tensors="pt",
            padding=True,
            return_offsets_mapping=False
        )
        
        input_ids = encodings['input_ids']
        attention_mask = encodings['attention_mask']
        
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=2)
            probs = torch.nn.functional.softmax(logits, dim=-1)
            confidence_scores_raw = torch.max(probs, dim=2).values
        
        pred_ids = predictions[0].cpu().numpy()
        conf_scores = confidence_scores_raw[0].cpu().numpy()
        token_ids = input_ids[0].tolist()
        
        word_id_map = {}
        current_token_pos = 1
        
        for word_idx, word in enumerate(tokens):
            word_tokens = tokenizer.tokenize(word)
            for sub_idx in range(len(word_tokens)):
                if current_token_pos < len(token_ids) - 1:
                    word_id_map[current_token_pos] = word_idx
                    current_token_pos += 1
        
        labels = []
        label_confidences = []
        debug_predictions = []
        used_words = set()
        
        for token_pos in range(1, min(len(token_ids) - 1, len(pred_ids))):
            if token_pos in word_id_map:
                word_idx = word_id_map[token_pos]
                if word_idx not in used_words:
                    pred_id = int(pred_ids[token_pos])
                    confidence = float(conf_scores[token_pos])
                    label = phobert_id2label.get(pred_id, "O")
                    
                    labels.append(label)
                    label_confidences.append(confidence)
                    
                    if debug:
                        debug_predictions.append({
                            "word": tokens[word_idx],
                            "label": label,
                            "confidence": confidence
                        })
                    
                    used_words.add(word_idx)
        
        if len(labels) > len(tokens):
            labels = labels[:len(tokens)]
            label_confidences = label_confidences[:len(tokens)]
        while len(labels) < len(tokens):
            labels.append("O")
            label_confidences.append(0.0)
        
        entities = extract_entities(tokens, labels, text, confidences=label_confidences)
        
        result = {
            "model": "PhoBERT",
            "text": text,
            "tokens": tokens,
            "labels": labels,
            "entities": entities,
            "confidence_scores": label_confidences,
            "avg_confidence": float(sum(label_confidences) / len(label_confidences)) if label_confidences else 0.0
        }
        
        if debug:
            result["debug_predictions"] = debug_predictions
        
        return result
    
    except Exception as e:
        print(f"Error in PhoBERT prediction: {e}")
        return {
            "model": "PhoBERT",
            "text": text,
            "tokens": tokenize(text),
            "labels": ["O"] * len(tokenize(text)),
            "entities": [],
            "error": str(e)
        }