import os
import urllib.request
import hashlib
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# We need GROQ_API_KEY from environment or .env
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    print("Error: GROQ_API_KEY not found in environment.")
    exit(1)

# Use the downloaded MP3 file
audio_path = "scratch/test_small.mp3"

# Read file bytes
with open(audio_path, "rb") as f:
    audio_bytes = f.read()

print(f"Audio file size: {len(audio_bytes)} bytes")

# Detect format from magic bytes
ext = "wav"
mime_type = "audio/wav"
if audio_bytes.startswith(b"\x1a\x45\xdf\xa3"):
    ext = "webm"
    mime_type = "audio/webm"
elif b"RIFF" in audio_bytes[:12]:
    ext = "wav"
    mime_type = "audio/wav"
elif b"ftyp" in audio_bytes[4:12]:
    ext = "m4a"
    mime_type = "audio/m4a"
elif audio_bytes.startswith(b"OggS"):
    ext = "ogg"
    mime_type = "audio/ogg"
elif audio_bytes.startswith(b"ID3") or audio_bytes.startswith(b"\xff\xfb") or audio_bytes.startswith(b"\xff\xf3"):
    ext = "mp3"
    mime_type = "audio/mp3"

print(f"Detected extension: {ext}, detected mime-type: {mime_type}")

# Call Groq Whisper API
try:
    client = Groq(api_key=GROQ_API_KEY)
    audio_name = f"audio.{ext}"
    print("Sending audio to Groq Whisper transcription API...")
    transcription = client.audio.transcriptions.create(
        file=(audio_name, audio_bytes, mime_type),
        model="whisper-large-v3-turbo"
    )
    user_prompt = transcription.text.strip()
    print(f"Transcription result: \"{user_prompt}\"")
except Exception as e:
    print(f"Failed to transcribe: {e}")
