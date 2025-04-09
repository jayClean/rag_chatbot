# RAG System Documentation

## Overview

This Retrieval-Augmented Generation (RAG) system allows you to ingest documents (PDFs and websites), store their content in a vector database, and ask questions about the ingested information. The system uses FastAPI for the backend, FAISS for vector storage, and Ollama for local LLM inference.

## Prerequisites

- Python 3.8+
- Ollama installed locally (for LLM inference)
- At least one model downloaded in Ollama (e.g., mistral)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/jayClean/rag_chatbot.git
   cd rag_chatbot
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Requirements

Create a `requirements.txt` file with the following dependencies:

```
fastapi==0.104.1
uvicorn==0.23.2
pydantic==2.4.2
python-multipart==0.0.6
requests==2.31.0
PyPDF2==1.26.0
beautifulsoup4==4.12.2
sentence-transformers==2.2.2
faiss-cpu==1.7.4
numpy==1.24.3
```

## Project Structure

```
rag-system/
├── main.py                # FastAPI application
├── rag_pipeline.py        # Core RAG functionality
├── embedder.py            # Text embedding functions
├── retriever.py           # Vector search functions
├── vector_store.py        # Vector store management
├── loaders/
│   ├── __init__.py
│   ├── pdf_loader.py      # PDF processing
│   └── web_scraper.py     # Website processing
├── vector_store/          # Directory for storing vectors (created automatically)
└── requirements.txt
```

## Running the Application

1. Make sure Ollama is running:
   ```bash
   ollama serve
   ```

2. Pull a model if you haven't already:
   ```bash
   ollama pull mistral
   ```

3. Start the FastAPI server:
   ```bash
   uvicorn main:app --reload
   ```

4. The API will be available at http://localhost:8000

## API Usage

### 1. Ingest a PDF Document

```bash
curl -X POST http://localhost:8000/ingest/pdf \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/your/document.pdf"
```

Response:
```json
{
  "message": "PDF ingested",
  "chunks": 10
}
```

### 2. Ingest a Website

```bash
curl -X POST http://localhost:8000/ingest/website \
  -H "Content-Type: application/json" \
  -d '{"url": "https://example.com"}'
```

Response:
```json
{
  "message": "Website ingested",
  "chunks": 15
}
```

### 3. Ask a Question

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What information is contained in the documents?"}'
```

Response:
```json
{
  "answer": "The documents contain information about..."
}
```

### 4. Check Vector Store Status

```bash
curl -X GET http://localhost:8000/status/vector-store
```

Response:
```json
{
  "vector_store_empty": false,
  "vector_count": 25
}
```

## Step-by-Step Usage Guide

### Step 1: Start the Services

1. Start Ollama:
   ```bash
   ollama serve
   ```

2. Start the RAG API:
   ```bash
   uvicorn main:app --reload
   ```

### Step 2: Ingest Documents

#### Option A: Using the API Directly

1. Ingest a PDF:
   ```bash
   curl -X POST http://localhost:8000/ingest/pdf \
     -H "Content-Type: multipart/form-data" \
     -F "file=@/path/to/your/document.pdf"
   ```

2. Ingest a website:
   ```bash
   curl -X POST http://localhost:8000/ingest/website \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com"}'
   ```

#### Option B: Using the Swagger UI

1. Open http://localhost:8000/docs in your browser
2. Navigate to the `/ingest/pdf` or `/ingest/website` endpoint
3. Click "Try it out"
4. Upload your PDF file or enter a website URL
5. Click "Execute"

### Step 3: Ask Questions

#### Option A: Using the API Directly

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the key points in the document?"}'
```

#### Option B: Using the Swagger UI

1. Open http://localhost:8000/docs in your browser
2. Navigate to the `/ask` endpoint
3. Click "Try it out"
4. Enter your question
5. Click "Execute"

### Step 4: Check Status (Optional)

To verify that documents have been ingested:

```bash
curl -X GET http://localhost:8000/status/vector-store
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - Solution: Pull the model using `ollama pull mistral`

2. **Timeout errors**:
   - Solution: Increase the timeout in `rag_pipeline.py` or try with a smaller document

3. **"No such file or directory" for vector store**:
   - Solution: The system will automatically create the directory on first use

4. **Empty responses from the LLM**:
   - Solution: Check that Ollama is running and the model is properly loaded

### Logs

Check the application logs for detailed error information. The system logs errors at various levels to help diagnose issues.

## Advanced Configuration

You can modify the following parameters in the code:

- `max_context_length` in `rag_pipeline.py`: Controls how much context is sent to the LLM
- `dim` in `retriever.py`: Embedding dimension (should match your embedding model)
- Models list in `rag_pipeline.py`: Change the list of models to try

## License

MIT

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.