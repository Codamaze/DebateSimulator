from fastapi import FastAPI, UploadFile, File, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from faster_whisper import WhisperModel
import tempfile
import os
from dotenv import load_dotenv
from pathlib import Path

# Load env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)
API_KEY = os.getenv("API_KEY")

# Auth
def require_api_key(x_api_key: str = Header(...)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized")

# Init FastAPI
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model once
model = WhisperModel("base", device="cpu", compute_type="int8")

@app.post("/transcribe", dependencies=[Depends(require_api_key)])
async def transcribe_audio(file: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        contents = await file.read()
        tmp.write(contents)
        tmp_path = tmp.name

    segments, _ = model.transcribe(tmp_path, beam_size=5)

    result_text = " ".join([segment.text.strip() for segment in segments])

    os.remove(tmp_path)

    return {"transcript": result_text.strip()}
