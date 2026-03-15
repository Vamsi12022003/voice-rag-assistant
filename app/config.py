"""
config.py - Centralised configuration
Automatically fixes SSL certificate path for Google Gemini SDK on Windows/Anaconda.
"""

import os
import ssl
import certifi
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── SSL Fix (must happen before any Google SDK import) ────────────────────────
# Anaconda on Windows often has a broken SSL store.
# Force Python and httpx to use certifi's trusted CA bundle.
_cert_path = certifi.where()
os.environ["SSL_CERT_FILE"]      = _cert_path
os.environ["REQUESTS_CA_BUNDLE"] = _cert_path
os.environ["GRPC_DEFAULT_SSL_ROOTS_FILE_PATH"] = _cert_path

# Patch the default SSL context globally
ssl._create_default_https_context = ssl.create_default_context


class Settings:
    GOOGLE_API_KEY:     str  = os.getenv("GEMINI_API_KEY", "")
    GROQ_API_KEY:       str  = os.getenv("GROQ_API_KEY", "")
    DATA_DIR:           str  = "data"
    VECTORSTORE_DIR:    str  = "vectorstore"
    AUDIO_UPLOAD_DIR:   str  = "audio_uploads"
    CHUNK_SIZE:         int  = 1000
    CHUNK_OVERLAP:      int  = 200
    MAX_RETRIEVED_DOCS: int  = 4
    LLM_MODEL:          str  = "llama-3.3-70b-versatile"
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    TTS_LANGUAGE:       str  = "en"
    TTS_SLOW:           bool = False

    def validate(self):
        if not self.GOOGLE_API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in .env file.")


settings = Settings()

# Flat constants
GOOGLE_API_KEY     = settings.GOOGLE_API_KEY
GEMINI_API_KEY     = settings.GOOGLE_API_KEY
DATA_DIR           = settings.DATA_DIR
VECTORSTORE_DIR    = settings.VECTORSTORE_DIR
AUDIO_UPLOAD_DIR   = settings.AUDIO_UPLOAD_DIR
CHUNK_SIZE         = settings.CHUNK_SIZE
CHUNK_OVERLAP      = settings.CHUNK_OVERLAP
MAX_RETRIEVED_DOCS = settings.MAX_RETRIEVED_DOCS
LLM_MODEL          = settings.LLM_MODEL
GROQ_API_KEY       = settings.GROQ_API_KEY
EMBEDDING_MODEL    = settings.EMBEDDING_MODEL

# Create required directories
for _dir in [DATA_DIR, VECTORSTORE_DIR, AUDIO_UPLOAD_DIR]:
    Path(_dir).mkdir(parents=True, exist_ok=True)