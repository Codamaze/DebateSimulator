from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Header, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
import os
from backend.debate_logic import DebateState, DebatePhase
from backend.llms_logic import generate_ai_speech, generate_judging_feedback
from local_audio.tts_utils import speak_aloud
from local_audio.cleaning_utils import clean_speech

# Load environment variables
load_dotenv(Path(__file__).resolve().parent / ".env")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
API_KEY = os.getenv("API_KEY")

# --- Security Dependency ---
def require_api_key(x_api_key: str = Header(...)):
    if not API_KEY or x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

debate_session = {"state": None}

@app.get("/")
async def root():
    return {"message": "✅ Public Forum Debate API is running."}

@app.get("/ping")
async def ping():
    return {"status": "ok"}

@app.post("/start-debate", dependencies=[Depends(require_api_key)])
async def start_debate(payload: dict):
    try:
        resolution = payload["resolution"]
        user_role = payload["user_role"].lower()
        ai_difficulty = payload["ai_difficulty"].lower()
        mode = payload.get("mode", "user-vs-ai")

        state = DebateState(resolution, user_role, ai_difficulty, mode)
        debate_session["state"] = state

        if user_role == "con":
            state.current_speaker = "ai"
            speech_type = state.get_expected_speech_type()
            ai_text = await generate_ai_speech(state, speech_type)
            state.log_speech("ai", state.ai_role, speech_type, ai_text)
            speak_aloud(clean_speech(ai_text))
            state.advance_phase()
        else:
            state.current_speaker = "user"

        return JSONResponse(content={"status": "initialized", "state": state.get_state()})
    except KeyError as e:
        return JSONResponse(status_code=400, content={"error": f"Missing key: {str(e)}"})

@app.websocket("/ws/debate")
async def debate_socket(websocket: WebSocket):
    token = websocket.headers.get("x-api-key")
    if token != API_KEY:
        await websocket.close(code=4401)  # Unauthorized
        return

    await websocket.accept()
    state: DebateState = debate_session.get("state")

    if not state:
        await websocket.send_json({"error": "No active debate session. Start via /start-debate"})
        await websocket.close()
        return

    await websocket.send_json({
        "status": "connected",
        "phase": state.current_phase.name,
        "speaker": state.current_speaker
    })

    try:
        while True:
            data = await websocket.receive_json()
            print("💬 Received:", data)

            if data["type"] == "end_phase":
                if state.current_phase == DebatePhase.FINAL_FOCUS_CON:
                    print("🎓 Judging Phase started.")
                    state.advance_phase()

                    await websocket.send_json({
                        "event": "phase_updated",
                        "state": state.get_state()
                    })

                    judging_result = await generate_judging_feedback(state)
                    await websocket.send_json({
                        "event": "judging_feedback",
                        "feedback": judging_result,
                        "state": state.get_state()
                    })
                    continue

                state.advance_phase()
                await websocket.send_json({
                    "event": "phase_updated",
                    "state": state.get_state()
                })

                if state.is_ai_turn():
                    speech_type = state.get_expected_speech_type()
                    ai_text = await generate_ai_speech(state, speech_type)
                    state.log_speech("ai", state.ai_role, speech_type, ai_text)

                    await websocket.send_json({
                        "event": "ai_speech",
                        "speech_type": speech_type,
                        "text": ai_text,
                        "state": state.get_state()
                    })
                continue

            if data["type"] != "speech":
                await websocket.send_json({"error": "Unsupported message type"})
                continue

            user_speech_type = data["speech_type"].lower()
            role = data["role"].lower()
            new_phase = map_speech_type_to_phase(user_speech_type, role, state.transcript)

            if not new_phase:
                await websocket.send_json({"error": f"Unknown speech_type: {user_speech_type}"})
                continue

            state.log_speech(
                speaker=data["speaker"],
                role=data["role"],
                speech_type=data["speech_type"],
                content=data["content"]
            )

            if user_speech_type == "crossfire":
                if state.current_phase != new_phase:
                    state.current_phase = new_phase
                    print(f"📌 Entering phase: {new_phase.name}")

                state.current_speaker = "ai" if data["speaker"] == "user" else "user"

                await websocket.send_json({
                    "event": "phase_updated",
                    "state": state.get_state()
                })

                if state.is_ai_turn():
                    print("🤖 AI (crossfire) responding...")
                    ai_text = await generate_ai_speech(state, "Crossfire")
                    state.log_speech("ai", state.ai_role, "Crossfire", ai_text)

                    await websocket.send_json({
                        "event": "ai_speech",
                        "speech_type": "Crossfire",
                        "text": ai_text,
                        "state": state.get_state()
                    })
                    speak_aloud(clean_speech(ai_text))
                continue

            state.current_phase = new_phase
            state.advance_phase()
            print(f"✅ Phase advanced → Now in: {state.current_phase.name}")

            await websocket.send_json({
                "event": "phase_updated",
                "state": state.get_state()
            })

            if state.is_ai_turn():
                ai_speech_type = state.get_expected_speech_type()
                print("🤖 AI turn —", ai_speech_type)
                ai_text = await generate_ai_speech(state, ai_speech_type)
                state.log_speech("ai", state.ai_role, ai_speech_type, ai_text)

                await websocket.send_json({
                    "event": "ai_speech",
                    "speech_type": ai_speech_type,
                    "text": ai_text,
                    "state": state.get_state()
                })
                speak_aloud(clean_speech(ai_text))

    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected.")

@app.get("/transcript", dependencies=[Depends(require_api_key)])
async def get_transcript():
    state = debate_session.get("state")
    if not state:
        return JSONResponse(status_code=400, content={"error": "No active debate."})
    return {"transcript": state.get_transcript()}


def map_speech_type_to_phase(speech_type: str, role: str, transcript: list):
    speech_type = speech_type.lower()
    speech_type = speech_type.strip().lower().replace("_", " ")
    role = role.lower()
    if speech_type == "constructive":
        return DebatePhase.PRO_CONSTRUCTIVE if role == "pro" else DebatePhase.CON_CONSTRUCTIVE
    elif speech_type == "rebuttal":
        return DebatePhase.PRO_REBUTTAL if role == "pro" else DebatePhase.CON_REBUTTAL
    elif speech_type == "summary":
        return DebatePhase.PRO_SUMMARY if role == "pro" else DebatePhase.CON_SUMMARY
    elif speech_type == "final focus":
        return DebatePhase.FINAL_FOCUS_PRO if role == "pro" else DebatePhase.FINAL_FOCUS_CON
    elif speech_type == "crossfire":
        crossfires = sum(1 for t in transcript if t["speech_type"].lower() == "crossfire")
        return DebatePhase.CROSSFIRE_2_QA if crossfires >= 4 else DebatePhase.CROSSFIRE_1_QA
    else:
        return None