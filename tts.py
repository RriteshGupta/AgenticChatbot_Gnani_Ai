import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("GNANI_API_KEY")

if not API_KEY:
    raise ValueError("GNANI_API_KEY is missing from .env")

url = "https://api.vachana.ai/api/v1/tts/inference"

def txt_to_speech(text):
    headers = {
    "X-API-Key-ID": API_KEY,
    "Content-Type": "application/json",
    }

    payload = {
    "text": text,
    "model": "timbre-v2.5",
    "voice": "Nalini",      
    "language": "hi-IN",
    "speed": 1.0,
    "audio_config": {
        "sample_rate": 48000,
        "num_channels": 1,
        "sample_width": 2,
        "encoding": "linear_pcm",
        "container": "wav",
         },
    }

    response = requests.post(url, headers= headers, json=payload, timeout=60)
    response.raise_for_status()
    return response.content

