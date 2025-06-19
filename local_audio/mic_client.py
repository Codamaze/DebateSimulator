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

            if speech_type.lower() != "crossfire":
                # Normal stage: just wait for one AI response
                while True:
                    try:
                        response = await ws.recv()
                        data = json.loads(response)
                        if data.get("event") == "ai_speech":
                            print(f"🤖 AI said ({data['speech_type']}):\n{data['text']}")
                            break
                    except Exception as e:
                        print("❌ Error reading from WebSocket:", e)
                        break

            else:
                # Crossfire stage: keep loop open until user says "stop"
                async def listen_ws():
                    try:
                        while True:
                            response = await ws.recv()
                            data = json.loads(response)
                            if data.get("event") == "ai_speech":
                                print(f"🤖 AI said ({data['speech_type']}):\n{data['text']}")
                            elif data.get("event") == "phase_updated":
                                # ✅ Only break if it's NOT a crossfire phase anymore
                                new_phase = data["state"]["phase"].lower()
                                if "crossfire" not in new_phase:
                                    print("✅ Phase ended.")
                                    break
                                else:
                                    # Still in crossfire — ignore this update
                                    continue
                            elif data.get("error"):
                                print(f"❌ Backend error: {data['error']}")
                                break
                    except Exception as e:
                        print("❌ WebSocket receive error:", e)

                async def user_input():
                    while True:
                        stop = await asyncio.get_event_loop().run_in_executor(None, input, "Type 'stop' to end crossfire: ")
                        if stop.strip().lower() == "stop":
                            await ws.send(json.dumps({"type": "end_phase"}))
                            print("🛑 Sent end_phase to backend.")
                            break

                await asyncio.gather(listen_ws(), user_input())

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
