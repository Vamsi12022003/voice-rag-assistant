"""
voice.py - Speech-to-Text (Whisper) and Text-to-Speech (gTTS)
"""

import os
import uuid
import whisper
import os
os.environ["PATH"] = r"C:\Users\krish\AppData\Local\Microsoft\WinGet\Packages\Gyan.FFmpeg_Microsoft.Winget.Source_8wekyb3d8bbwe\ffmpeg-8.0.1-full_build\bin" + os.pathsep + os.environ["PATH"]
from gtts import gTTS
from app.config import AUDIO_UPLOAD_DIR

print("Loading Whisper model...")
whisper_model = whisper.load_model("base")
print("Whisper model loaded.")


def transcribe_audio(audio_file_path: str) -> str:
    """Convert audio file to text using local Whisper model."""
    result = whisper_model.transcribe(audio_file_path)
    return result["text"].strip()


def text_to_speech(text: str) -> str:
    """Convert text to speech using gTTS. Returns the saved file path."""
    filename    = f"{uuid.uuid4()}.mp3"
    output_path = os.path.join(AUDIO_UPLOAD_DIR, filename)
    tts         = gTTS(text=text, lang="en", slow=False)
    tts.save(output_path)
    return output_path


def save_upload(file_bytes: bytes, extension: str = "wav") -> str:
    """Save uploaded audio bytes to disk and return the file path."""
    filename  = f"{uuid.uuid4()}.{extension}"
    file_path = os.path.join(AUDIO_UPLOAD_DIR, filename)
    with open(file_path, "wb") as f:
        f.write(file_bytes)
    return file_path