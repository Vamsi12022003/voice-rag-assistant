# Voice-Enabled Medical RAG Assistant

A production-ready AI assistant that lets healthcare professionals query medical documents by voice or text — and get accurate, source-cited answers in seconds.

Built for environments where speed matters. Tested on MSF Clinical Guidelines (100 pages, 253 chunks).

**[▶ Watch Demo](https://drive.google.com/file/d/1NDDPdhT35RaXfVTBzhwFRURnj_U8hR77/view)**

---

## What it does

A nurse asks: *"What is the paracetamol dose for a 10kg child with fever?"*  
The assistant retrieves the exact dosage from page 36 of the clinical guidelines and speaks the answer aloud — in under 3 seconds.

- **Voice input** — speak your query, get a spoken answer (Whisper STT + gTTS TTS)
- **Text input** — type queries via REST API
- **RAG pipeline** — answers grounded in your PDF knowledge base, never hallucinated
- **Source citations** — every answer includes exact filename + page number
- **Conversation memory** — context preserved across turns per session
- **Fully containerised** — Docker + docker-compose, one command to run

---

## Stack

| Component | Technology |
|---|---|
| LLM | Groq `llama-3.3-70b-versatile` |
| Embeddings | HuggingFace `all-MiniLM-L6-v2` (local, no API) |
| Vector Store | FAISS (local) |
| Framework | LangChain |
| API | FastAPI + Uvicorn |
| STT | OpenAI Whisper base (local) |
| TTS | gTTS |
| Database | SQLite via SQLAlchemy |
| Container | Docker + docker-compose |

---

## Quick Start

### With Docker (recommended)

```bash
git clone https://github.com/Vamsi12022003/voice-rag-assistant.git
cd voice-rag-assistant
cp .env.example .env          # add your GROQ_API_KEY
docker-compose up --build
```

### Without Docker

```bash
conda create -n voicerag python=3.12.7 && conda activate voicerag
pip install -r requirements.txt
cp .env.example .env          # add your GROQ_API_KEY
curl -X POST http://127.0.0.1:8000/ingest
uvicorn app.main:app --reload
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Health check |
| POST | `/ingest` | Build FAISS index from PDFs in `data/` |
| POST | `/ask` | Text query → JSON answer + citations |
| POST | `/ask/voice` | Voice query (WAV) → JSON + MP3 response |
| GET | `/history/{session_id}` | Retrieve conversation history |
| POST | `/clear/{session_id}` | Reset session memory |

### Example

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the paracetamol dose for children with fever?", "session_id": "demo1"}'
```

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

---

## Knowledge Base

Currently ingested: MSF Clinical Guidelines (100 pages, 253 chunks).  
Drop any medical PDF into `data/` and run `POST /ingest` to add it.

---

## Known Limitations

- Whisper `base` misrecognises medical terms (e.g. "paracetamol" → "parasitamol") — upgrading to `whisper-small` improves accuracy
- `chunk_size=1000` can split protocols mid-table — reducing to 600 with `overlap=150` is a planned fix
- Docker not tested on Windows — use local setup on Windows

---

## Author

**P. Krishna Vamsi** — [GitHub](https://github.com/Vamsi12022003) · [LinkedIn](your-linkedin-url)
