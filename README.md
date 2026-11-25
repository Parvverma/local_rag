# Local Multimodal PDF RAG

A completely local, privacy-preserving Retrieval-Augmented Generation (RAG) system for querying PDF documents—including text, scanned pages, diagrams, charts, and tables—using **Ollama** and **ChromaDB**.

## About the Project

This project lets you index local PDF files and ask natural-language questions from your terminal without sending documents to external APIs.

### How It Works

1. **Ingestion (`ingest.py`)**
   - Reads all `.pdf` files from the `documents/` directory.
   - Extracts digital text with **PyMuPDF** and automatically runs **Tesseract OCR** (`dpi=250`) on pages with little digital text (`< 200` characters) or embedded images.
   - Optionally renders each page as an image (`ENABLE_VISION = True`) and uses **Gemma (`gemma4:e4b`)** to generate searchable descriptions of diagrams, flowcharts, graphs, tables, and technical illustrations.
   - Splits extracted text and visual descriptions into overlapping chunks (`CHUNK_SIZE = 1000`, `CHUNK_OVERLAP = 200`), generates embeddings with **`embeddinggemma`**, and stores them locally in **ChromaDB** (`./chroma_db`, collection `pdf_knowledge`).

2. **Question Answering (`chat.py`)**
   - Expands your question into up to 3 alternative technical search queries using `gemma4:e4b`.
   - Retrieves and deduplicates matching chunks from ChromaDB, keeping results within `MAX_DISTANCE` (default `0.7`).
   - Uses the LLM (`format="json"`) to rerank retrieved chunks by relevance (with automatic fallback to vector similarity order).
   - Generates a grounded answer with document name and page number citations.

## Project Structure

```text
local-rag/
├── ingest.py          # PDF extraction, OCR, vision analysis, chunking, and indexing
├── chat.py            # Interactive RAG CLI with query expansion and reranking
├── requirements.txt   # Pinned Python dependencies
├── .gitignore         # Excludes venv, local DB, PDFs, and OS/editor artifacts
└── documents/         # Place your input PDF files here (ignored by Git)
```

## Libraries & Dependencies

| Package | Version | Purpose |
| --- | --- | --- |
| `pymupdf` | `2.1` | PDF parsing, page rendering, and Tesseract OCR integration |
| `ollama` | `0.6.3` | Local LLM chat, vision analysis, and embedding generation |
| `chromadb` | `1.5.9` | Persistent local vector database |
| `pillow` | `13.0.0` | Image handling support |

### System Prerequisites

- **Python 3.10+** (tested on Python 3.14.2)
- **[Ollama](https://ollama.com/)** installed and running locally
- **Tesseract OCR** installed and available in your `PATH` (used by `ingest.py` for scanned pages or pages containing images):
  - **macOS (Homebrew):**
    ```bash
    brew install tesseract
    ```
  - **Ubuntu / Debian:**
    ```bash
    sudo apt update && sudo apt install -y tesseract-ocr
    ```
  - **Windows:** Install from [UB-Mannheim Tesseract releases](https://github.com/UB-Mannheim/tesseract/wiki) and add the install folder to `PATH`.

## Setup Instructions

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd local-rag
```

### 2. Create and Activate a Virtual Environment

**macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Python Dependencies

Install from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Or install all pinned packages directly:

```bash
pip install pymupdf==2.1 ollama==0.6.3 chromadb==1.5.9 pillow==13.0.0
```

### 4. Pull the Required Ollama Models

Make sure Ollama is running (start it with `ollama serve` if it is not already running as a background service), then pull the LLM and embedding models:

```bash
ollama pull gemma4:e4b
ollama pull embeddinggemma
```

## Usage

### Step 1: Add PDFs

Create the `documents/` directory if it does not exist and place one or more `.pdf` files inside it:

```bash
mkdir -p documents
cp /path/to/your-file.pdf documents/
```

### Step 2: Ingest Documents

Run the ingestion script to extract text, run OCR/vision analysis, generate embeddings, and store chunks in `./chroma_db`:

```bash
python ingest.py
```

> **Tip:** To skip vision analysis on every page and speed up ingestion, set `ENABLE_VISION = False` in `ingest.py`.

### Step 3: Chat with Your Documents

Start the interactive chatbot:

```bash
python chat.py
```

Filter answers to a specific PDF filename:

```bash
python chat.py --source your-file.pdf
```

Type `exit` or `quit` (or press `Ctrl+C`) to stop.

### CLI Options & Environment Variables (`chat.py`)

`chat.py` accepts the following CLI flags (and matching `RAG_*` environment variables):

| Flag | Env Variable | Default | Description |
| --- | --- | --- | --- |
| `--source` | — | `None` | Filter retrieval to a single PDF filename (e.g. `notes.pdf`) |
| `--db-dir` | `RAG_DB_DIR` | `./chroma_db` | ChromaDB persistence directory |
| `--collection` | `RAG_COLLECTION_NAME` | `pdf_knowledge` | ChromaDB collection name |
| `--model` | `RAG_LLM_MODEL` | `gemma4:e4b` | Ollama LLM model for expansion, reranking, and answers |
| `--embedding-model` | `RAG_EMBEDDING_MODEL` | `embeddinggemma` | Ollama embedding model |
| `--top-k` | `RAG_TOP_K` | `5` | Number of chunks retrieved per query and kept after reranking |
| `--max-distance` | `RAG_MAX_DISTANCE` | `0.7` | Maximum embedding distance threshold |

Run `python chat.py --help` to view all options.
