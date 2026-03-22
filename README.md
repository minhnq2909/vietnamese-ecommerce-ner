# vietnamese-ecommerce-ner

AI-powered Named Entity Recognition system for Vietnamese text. Identifies product names, types, prices, and locations using two models: **PhoBERT** (transformer) and **BiLSTM-CRF**.

## Features

- Recognizes 4 entity types: PRODUCT_NAME, PRODUCT_TYPE, PRICE, LOCATION
- Supports 2 models: **PhoBERT** and **BiLSTM-CRF**
- Supports price formats: 2m6, 2tr5, 4k5, 3 triệu, 5000 đ
- Modern web interface with light theme
- RESTful API for integration

## Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/minhnq2909/vietnamese-ecommerce-ner.git
cd vietnamese-ecommerce-ner
```

### 2. Install Python Dependencies

The project uses a unified `requirements.txt` in the root directory that covers all components:

- Backend API (Flask)
- Training scripts (PyTorch, Transformers)
- Jupyter notebooks
- All source code dependencies

**Option A: Using Docker (Recommended)**
Dependencies are automatically installed when building Docker images:

```bash
docker-compose build
```

**Option B: Local Installation**
Create and activate a Python 3.10+ virtual environment, then install:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies from root requirements.txt
pip install -r requirements.txt
```

After installation, you can run:

- Backend: `python backend/app.py`
- Training: `python src/train_lstm.py` or `python src/train_phobert.py`
- Notebooks: `jupyter notebook notebooks/`

### 3. Download Required Models & Dependencies

**VnCoreNLP** (for Vietnamese word segmentation):

```bash
cd backend
mkdir -p vncorenlp
cd vncorenlp

# Download VnCoreNLP jar and models
wget https://github.com/vncorenlp/VnCoreNLP/releases/download/v1.1.1/VnCoreNLP-1.1.1.jar
mkdir -p models/{ner,postagger,wordsegmenter,dep}

# Download specific models
wget -P models/ner https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/ner/ner.mdl
wget -P models/postagger https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/postagger/postagger.mdl
wget -P models/wordsegmenter https://raw.githubusercontent.com/vncorenlp/VnCoreNLP/master/models/wordsegmenter/wordsegmenter.mdl

cd ..
```

**AI Models** (download from Google Drive):

1. **Access the Google Drive folder**:
   - Open https://drive.google.com/drive/folders/1izc-BFRhn2Z7ZTaKRUDa5z1KlgeiNiJO
   - You'll see 2 folders: `bilstm/` and `phobert/`

2. **Download the files**
   - Download the `bilstm/` folder from Drive
   - Download the `phobert/` folder from Drive
   - Extract to `backend/artifacts/`

3. **Verify the structure** in `backend/artifacts/`:

```bash
# Structure of AI model:
backend/
└── artifacts/
    ├── phobert/               # PhoBERT model folder
    │   ├── config.json
    │   ├── tokenizer_config.json
    │   ├── model.safetensors
    │   ├── vocab.txt
    │   ├── added_tokens.json
    │   └── bpe.codes
    └── bilstm/                # BiLSTM-CRF model folder
        ├── bilstm_crf.pt      # BiLSTM-CRF trained model
        ├── word2idx.pkl       # Word to index mapping
        └── idx2tag.pkl        # Index to tag mapping
```

### 4. Build and Start Services

```bash
cd ../..  # Back to project root

# Build Docker images
docker-compose build

# Start services
docker-compose up -d
docker-compose ps
```

### 5. Access Application

- **Web UI**: http://localhost:8080
- **API**: http://localhost:5000

## Usage

### Web Interface

Enter Vietnamese text in the input field. Both models will automatically analyze and highlight entities with color-coded labels.

### API Example

```bash
curl -X POST http://localhost:5000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Bán iPhone 15 giá 30tr tại Hà Nội"}'
```

## Common Commands

```bash
# Stop services
docker-compose down

# Restart after code changes
docker-compose build
docker-compose up -d

# View logs
docker-compose logs -f backend

# Clean everything
docker-compose down -v
```

## Troubleshooting

| Issue                     | Solution                                            |
| ------------------------- | --------------------------------------------------- |
| Port already in use       | `docker-compose down -v` then `up -d` again         |
| Models not loading        | `docker-compose build backend`                      |
| Network error on frontend | Rebuild: `docker-compose down -v && up -d`          |
| Changes not applied       | Use `--no-cache`: `docker-compose build --no-cache` |

## License

MIT - Free for academic and commercial use
