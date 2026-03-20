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

### 2. Download Required Models & Dependencies

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

Download files from the project's Google Drive link and add to `backend/artifacts/`:

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

### 3. Build and Start Services

```bash
cd ../..  # Back to project root

# Build Docker images
docker-compose build

# Start services
docker-compose up -d
docker-compose ps
```

### 4. Access Application

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
| Models not loading        | `docker-compose build backend`           |
| Network error on frontend | Rebuild: `docker-compose down -v && up -d`          |
| Changes not applied       | Use `--no-cache`: `docker-compose build --no-cache` |

## License

MIT - Free for academic and commercial use
