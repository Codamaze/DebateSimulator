import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
import asyncio
import json
import websockets
import httpx
import os
import sys
from dotenv import load_dotenv
from websockets.http import Headers
from websockets.legacy.client import connect

# === Load Environment ===
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))
API_KEY = os.getenv("API_KEY")

# === Constants ===
TRANSCRIBE_URL = "http://localhost:8000/transcribe"
WEBSOCKET_URL = "ws://localhost:8001/ws/debate"
SAMPLE_RATE = 16000

# === Audio Recorder ===
def record_audio():
    print("🎙️ Press ENTER to start speaking...")
    input()
    print("🔴 Recording... Press ENTER again to stop.")

    recording = []

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ Status: {status}")
        recording.append(indata.copy())

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
        input()  # Wait for ENTER to stop
        print("🛑 Stopped recording.")

    audio_np = np.concatenate(recording, axis=0).flatten()
    return audio_np

# === Transcription ===
async def transcribe_audio(audio_np):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        write(tmpfile.name, SAMPLE_RATE, audio_np)
        with open(tmpfile.name, "rb") as f:
            files = {"file": ("audio.wav", f, "audio/wav")}
            headers = {"x-api-key": API_KEY}
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.post(TRANSCRIBE_URL, files=files, headers=headers)
                    if response.status_code == 200:
                        transcript = response.json()["transcript"]
                        print("📝 Transcription:", transcript)
                        return transcript
                    else:
                        print("❌ Transcription error:", response.text)
                        return None
                except Exception as e:
                    print("❌ HTTP Error:", e)
                    return None

# === Send to Debate Backend ===
async def send_to_backend(transcript: str, role: str, speech_type: str):
    uri = WEBSOCKET_URL

    # Prepare headers for legacy client
    headers = [("x-api-key", API_KEY)]

    try:
        async with connect(uri, extra_headers=headers) as ws:
            await ws.send(json.dumps({
                "type": "speech",
                "role": role.lower(),
                "speaker": "user",
                "speech_type": speech_type,
                "content": transcript
            }))
            print("📤 Sent transcript to backend.")
    except Exception as e:
        print("❌ WebSocket error:", e)


# === Main CLI Entrypoint ===
async def main():
    if len(sys.argv) < 3:
        print("Usage: python mic_client.py <role> <speech_type>")
        return

    role = sys.argv[1]
    speech_type = sys.argv[2]

    audio_np = record_audio()
    transcript = await transcribe_audio(audio_np)

    if transcript:
        await send_to_backend(transcript, role, speech_type)

if __name__ == "__main__":
    asyncio.run(main())
