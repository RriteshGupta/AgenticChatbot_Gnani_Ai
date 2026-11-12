import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GNANI_API_KEY")

if not API_KEY:
    raise ValueError("GNANI_API_KEY is missing from .env")

url = "https://api.vachana.ai/stt/v3"

# Gnani STT REST: audio 30 second tak, WAV/MP3/OGG/FLAC/AAC/M4A
MAX_SECONDS = 30


def _error_message(response):
    """Gnani ke alag-alag error formats se padhne layak message nikalta hai."""
    try:
        body = response.json()
    except ValueError:
        return f"HTTP {response.status_code}"

    # {"success": false, "error": {"type": ..., "message": ...}}
    if isinstance(body.get("error"), dict):
        err = body["error"]
        return f"{err.get('type', 'ERROR')}: {err.get('message', '')}"

    # {"detail": {"error_code": ..., "message": ...}} (auth / rate limit)
    detail = body.get("detail")
    if isinstance(detail, dict):
        return f"{detail.get('error_code', 'ERROR')}: {detail.get('message', '')}"

    # {"detail": [...]} (validation, jaise language_code missing)
    if isinstance(detail, list) and detail:
        return "; ".join(str(d.get("msg", d)) for d in detail)

    return f"HTTP {response.status_code}: {body}"


def speech_to_text(audio_bytes, language_code="hi-IN", filename="recording.wav"):
    """Audio bytes ko Gnani Vachana STT se text mein badalta hai.

    audio_bytes   : WAV/MP3/OGG/FLAC/AAC/M4A ka raw content (max 30 sec)
    language_code : hi-IN, en-IN, ta-IN, te-IN, bn-IN, mr-IN, gu-IN, kn-IN, ml-IN, pa-IN
    Lautata hai  : transcript (str). Awaaz na ho to khaali string "".
    """
    if not audio_bytes:
        return ""

    headers = {"X-API-Key-ID": API_KEY}
    files = {"audio_file": (filename, audio_bytes, "audio/wav")}
    data = {
        "language_code": language_code,
        "format": "transcribe",  # numbers/dates ko likhit roop mein (ITN)
    }

    response = requests.post(url, headers=headers, files=files, data=data, timeout=60)

    if not response.ok:
        raise RuntimeError(_error_message(response))

    body = response.json()
    return (body.get("transcript") or "").strip()
