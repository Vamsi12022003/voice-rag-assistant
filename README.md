# Voice-Enabled Medical RAG Assistant

A production-grade AI assistant that lets healthcare professionals query medical documents using voice or text. Built with RAG (Retrieval-Augmented Generation) for accurate, source-cited answers.

## Demo

> Add demo video link here after recording

## Features

- **Voice input** — speak your query, get a spoken answer (Whisper STT + gTTS TTS)
- **Text input** — type queries via REST API
- **RAG pipeline** — answers grounded in your PDF knowledge base
- **Source citations** — every answer includes exact file + page number
- **Conversation memory** — context preserved across turns per session
- **REST API** — fully documented FastAPI endpoints

## Stack

| Component | Technology |
|-----------|-----------|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace all-MiniLM-L6-v2 (local) |
| Vector Store | FAISS (local) |
| Framework | LangChain |
| API | FastAPI + Uvicorn |
| STT | OpenAI Whisper (base, local) |
| TTS | gTTS |
| Database | SQLite via SQLAlchemy |
| Container | Docker + docker-compose |

## Quick Start

### Without Docker

```bash
# Clone repo
git clone https://github.com/Vamsi12022003/voice-rag-assistant.git
cd voice-rag-assistant

# Create env and install deps
conda create -n voicerag python=3.12.7
conda activate voicerag
pip install -r requirements.txt

# Add API keys
cp .env.example .env
# Edit .env with your GROQ_API_KEY

# Ingest documents
curl -X POST http://127.0.0.1:8000/ingest

# Start server
uvicorn app.main:app --reload
```

### With Docker

```bash
cp .env.example .env
# Edit .env with your GROQ_API_KEY
docker-compose up --build
```

## API Endpoints

### Health Check
```bash
GET /
```

### Ingest Documents
```bash
POST /ingest
```
Builds FAISS index from PDFs in `data/` folder.

### Text Query
```bash
POST /ask
Content-Type: application/json

{
  "question": "What is the paracetamol dose for children with fever?",
  "session_id": "demo1"
}
```

### Voice Query
```bash
POST /ask/voice
Content-Type: multipart/form-data

file: <wav file>
session_id: demo1
```

### Chat History
```bash
GET /history/{session_id}
```

### Clear Session
```bash
POST /clear/{session_id}
```

## Example Responses

**Text query:**
```json
{
  "answer": "Children 1 month and over: 15 mg/kg 3 to 4 times daily (max. 60 mg/kg daily)",
  "sources": [
    {"document": "medical_guidelines.pdf", "page": 36},
    {"document": "medical_guidelines.pdf", "page": 41}
  ],
  "session_id": "demo1"
}
```

**Voice query** returns same JSON plus `audio_response` path to MP3 file.

## Knowledge Base

Currently ingested: MSF Clinical Guidelines (100 pages, 253 chunks)

Drop any medical PDF into `data/` and hit `/ingest` to add it.

## Known Limitations

- Whisper base model may misrecognise medical terminology (e.g. "paracetamol" → "parasitamol"). Upgrading to whisper-small improves accuracy.
- Docker not tested on Windows — use local setup on Windows.

## Author

P. Krishna Vamsi — [GitHub](https://github.com/Vamsi12022003) | [LinkedIn](#)
