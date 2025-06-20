import sounddevice as sd
from scipy.io.wavfile import write
import tempfile
import numpy as np
import asyncio
import json
import httpx
import os
import sys
from dotenv import load_dotenv
from websockets.legacy.client import connect
import websockets
import threading

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
    stop_recording = threading.Event()

    def callback(indata, frames, time, status):
        if status:
            print(f"⚠️ Status: {status}")
        recording.append(indata.copy())

    def wait_for_enter():
        input()  # Waits for user to press ENTER again
        stop_recording.set()

    # Start the thread to watch for ENTER press
    enter_thread = threading.Thread(target=wait_for_enter)
    enter_thread.start()

    with sd.InputStream(samplerate=SAMPLE_RATE, channels=1, dtype='int16', callback=callback):
        while not stop_recording.is_set():
            sd.sleep(100)  # Avoids blocking the stream

    print("🛑 Stopped recording.")
    return np.concatenate(recording, axis=0).flatten()

# === Transcription ===
from pathlib import Path

async def transcribe_audio(audio_np):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
            write(tmpfile.name, SAMPLE_RATE, audio_np)
            tmpfile.flush()  # ensure all data is written

            tmp_path = Path(tmpfile.name)

        for attempt in range(2):  # Try twice
            try:
                with open(tmp_path, "rb") as f:
                    files = {"file": ("audio.wav", f, "audio/wav")}
                    headers = {"x-api-key": API_KEY}
                    timeout = httpx.Timeout(600.0, connect=30.0)

                    async with httpx.AsyncClient(timeout=timeout) as client:
                        response = await client.post(TRANSCRIBE_URL, files=files, headers=headers)
                        if response.status_code == 200:
                            transcript = response.json()["transcript"]
                            print("📝 Transcription:", transcript)
                            break  # success
                        else:
                            print(f"❌ Transcription error {response.status_code}: {response.text}")
                            transcript = None
            except (httpx.TimeoutException, httpx.RequestError) as e:
                print(f"⚠️ Attempt {attempt+1} failed: {e}. Retrying...")
                await asyncio.sleep(1)
                continue
            except Exception as e:
                print(f"❌ Unexpected error: {e}")
                transcript = None
                break

        # 🔐 Cleanup (even if it failed)
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception as e:
            print(f"⚠️ Failed to delete temp file: {e}")

        return transcript

    except Exception as outer_error:
        print(f"❌ Outer error in transcription: {outer_error}")
        return None



# === Debate WebSocket Handler ===
async def send_to_backend(role: str):
    headers = [("x-api-key", API_KEY)]

    try:
        async with connect(
            WEBSOCKET_URL,
            extra_headers=headers,
            ping_interval=1,
            ping_timeout=3600
        ) as ws:
            print("✅ Connected to debate backend.")

            while True:
                print("\n🎤 Type your speech type (constructive, rebuttal, summary, crossfire, final_focus) or 'stop' to exit:")
                speech_type = await asyncio.get_event_loop().run_in_executor(None, input)
                speech_type = speech_type.strip().lower()

                if speech_type == "stop":
                    print("👋 Exiting.")
                    break
                if speech_type.strip().lower() == "end":
                    print("🛎️ Sending end_phase to trigger judging...")
                    await ws.send(json.dumps({"type": "end_phase"}))

                    while True:
                        response = await ws.recv()
                        data = json.loads(response)
                        if data.get("event") == "judging_feedback":
                            print(f"\n🏁 Judging Feedback:\n{data['feedback']}")
                            return
                        elif data.get("error"):
                            print(f"❌ Backend error: {data['error']}")
                            return

                    return

                                    
                # === Handle Crossfire Mode ===
                if speech_type == "crossfire":
                    while True:
                        print("\n🎤 Press ENTER to ask a crossfire question, or type 'stop' to end crossfire:")
                        user_input = await asyncio.get_event_loop().run_in_executor(None, input)

                        if user_input.strip().lower() == "stop":
                            await ws.send(json.dumps({"type": "stop_crossfire"}))
                            print("🛑 Sent stop_crossfire to backend.")
                            break
                        elif user_input.strip() != "":
                            print("⚠️ Invalid input during crossfire. Press ENTER to speak or type 'stop'.")
                            continue

                        audio_np = record_audio()
                        transcript = await transcribe_audio(audio_np)
                        if not transcript:
                            print("❌ Skipping due to failed transcription.")
                            continue

                        await ws.send(json.dumps({
                            "type": "speech",
                            "role": role.lower(),
                            "speaker": "user",
                            "speech_type": "crossfire",
                            "content": transcript
                        }))
                        print("📤 Sent crossfire question to backend.")

                        # Wait for AI response
                        try:
                            while True:
                                response = await ws.recv()
                                data = json.loads(response)

                                if data.get("event") == "ai_speech":
                                    print(f"\n🤖 AI said ({data['speech_type']}):\n{data['text']}")
                                    break
                                elif data.get("event") == "crossfire_ended":
                                    print("✅ Crossfire ended by system.")
                                    break
                                elif data.get("error"):
                                    print(f"❌ Backend error: {data['error']}")
                                    break
                        except websockets.exceptions.ConnectionClosed:
                            print("🔌 Connection closed by server.")
                            return
                    continue

                # === Normal Speech Types ===
                audio_np = record_audio()
                transcript = await transcribe_audio(audio_np)

                if not transcript:
                    print("❌ Skipping due to failed transcription.")
                    continue
                print("sending to backend")
                
                await ws.send(json.dumps({
                    "type": "speech",
                    "role": role.lower(),
                    "speaker": "user",
                    "speech_type": speech_type,
                    "content": transcript
                }))
                print("📤 Sent to backend.")

                try:
                    while True:
                        response = await ws.recv()
                        data = json.loads(response)

                        if data.get("event") == "ai_speech":
                            print(f"\n🤖 AI said ({data['speech_type']}):\n{data['text']}")
                            break
                        elif data.get("event") == "judging_feedback":
                            print(f"\n🏁 Judging Feedback:\n{data['feedback']}")
                            break
                        elif data.get("error"):
                            print(f"❌ Backend error: {data['error']}")
                            break
                except websockets.exceptions.ConnectionClosed:
                    print("🔌 Connection closed by server.")
                    return

    except Exception as e:
        print("❌ WebSocket connection error:", e)

# === Entrypoint ===
async def main():
    if len(sys.argv) < 2:
        print("Usage: python mic_client.py <role>")
        print("Example: python mic_client.py pro")
        return
    role = sys.argv[1]
    await send_to_backend(role)

if __name__ == "__main__":
    asyncio.run(main())
