"""
main.py - FastAPI application entry point
Voice-Enabled Medical RAG Assistant
"""

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
import json

from app.rag import rag_pipeline
from app.voice import transcribe_audio, text_to_speech, save_upload
from app.database import init_db, save_message, get_chat_history

app = FastAPI(
    title="Voice-Enabled Medical RAG Assistant",
    description="Ask medical questions via text or voice. Get cited answers.",
    version="1.0.0",
)


@app.on_event("startup")
async def startup_event():
    init_db()                  # create SQLite tables
    rag_pipeline.initialize()  # load FAISS index + warm up LLM


class TextQuery(BaseModel):
    question:   str
    session_id: str = "default"


@app.get("/")
def root():
    return {"status": "running", "message": "Voice RAG Assistant is live"}


@app.post("/ingest")
def ingest_documents():
    """Build/rebuild the FAISS vector store from PDFs in data/."""
    try:
        count = rag_pipeline.ingest_documents()
        return {"message": "Successfully ingested documents", "chunks": count}
    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask")
def ask_question(query: TextQuery):
    """Answer a text question using RAG with conversation memory."""
    try:
        result      = rag_pipeline.query(query.question, query.session_id)
        sources_str = json.dumps(result["sources"])
        save_message(query.session_id, "user",      query.question)
        save_message(query.session_id, "assistant", result["answer"], sources_str)
        return {
            "answer":     result["answer"],
            "sources":    result["sources"],
            "session_id": query.session_id,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask/voice")
async def ask_voice(file: UploadFile = File(...), session_id: str = "default"):
    """Upload audio, get transcription + RAG answer + TTS audio response."""
    try:
        audio_bytes         = await file.read()
        ext                 = file.filename.split(".")[-1] if file.filename else "wav"
        audio_path          = save_upload(audio_bytes, ext)
        question            = transcribe_audio(audio_path)
        result              = rag_pipeline.query(question, session_id)
        sources_str         = json.dumps(result["sources"])
        audio_response_path = text_to_speech(result["answer"])
        save_message(session_id, "user",      question)
        save_message(session_id, "assistant", result["answer"], sources_str)
        return {
            "question":       question,
            "answer":         result["answer"],
            "sources":        result["sources"],
            "audio_response": audio_response_path,
            "session_id":     session_id,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/audio/{filename}")
def get_audio(filename: str):
    """Serve a generated TTS audio file."""
    return FileResponse(f"audio_uploads/{filename}", media_type="audio/mpeg")


@app.get("/history/{session_id}")
def get_history(session_id: str):
    """Return full chat history for a session from SQLite."""
    messages = get_chat_history(session_id)
    return {
        "session_id": session_id,
        "messages": [
            {
                "role":      m.role,
                "message":   m.message,
                "sources":   json.loads(m.sources) if m.sources else [],
                "timestamp": str(m.timestamp),
            }
            for m in messages
        ],
    }


@app.post("/clear/{session_id}")
def clear_memory(session_id: str):
    """Clear in-memory conversation history for a session."""
    rag_pipeline.clear_memory(session_id)
    return {"message": f"Memory cleared for session {session_id}"}