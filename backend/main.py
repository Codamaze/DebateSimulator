from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from pathlib import Path
import os
from backend.debate_logic import DebateState, DebatePhase
from backend.llms_logic import generate_ai_speech

# Load .env
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

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

@app.post("/start-debate")
async def start_debate(payload: dict):
    try:
        resolution = payload["resolution"]
        user_role = payload["user_role"].lower()
        ai_difficulty = payload["ai_difficulty"].lower()
        mode = payload.get("mode", "user-vs-ai")

        state = DebateState(resolution, user_role, ai_difficulty, mode)
        debate_session["state"] = state

        return JSONResponse(content={"status": "initialized", "state": state.get_state()})

    except KeyError as e:
        return JSONResponse(status_code=400, content={"error": f"Missing key: {str(e)}"})

@app.get("/ai-test")
async def ai_test():
    state = debate_session.get("state")
    if not state:
        return {"error": "No active session"}
    result = await generate_ai_speech(state, "Constructive")
    return {"result": result}


# Utility function: map speech_type → DebatePhase
def map_speech_type_to_phase(speech_type: str, role: str, transcript: list):
    speech_type = speech_type.lower()
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
        crossfire_count = sum(1 for t in transcript if t["speech_type"].lower() == "crossfire")
        return DebatePhase.CROSSFIRE_2_QA if crossfire_count >= 4 else DebatePhase.CROSSFIRE_1_QA
    else:
        return None


@app.websocket("/ws/debate")
async def debate_socket(websocket: WebSocket):
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

            if data["type"] != "speech":
                await websocket.send_json({"error": "Unsupported message type"})
                continue

            speech_type = data["speech_type"]
            role = data["role"]

            new_phase = map_speech_type_to_phase(speech_type, role, state.transcript)
            if not new_phase:
                await websocket.send_json({"error": f"Unknown speech_type: {speech_type}"})
                continue

            # Log user speech
            state.log_speech(
                speaker=data["speaker"],
                role=data["role"],
                speech_type=speech_type,
                content=data["content"]
            )

            if speech_type.lower() == "crossfire":
             # Update phase if needed
                if state.current_phase != new_phase:
                    state.current_phase = new_phase
                    print(f"📌 Entering phase: {new_phase.name}")

                # Toggle speaker
                state.current_speaker = "ai" if state.current_speaker == "user" else "user"
                print("🔁 Crossfire turn switched — speaker:", state.current_speaker)

                # Send update after switch
                await websocket.send_json({
                "event": "phase_updated",
                "state": state.get_state()
                })

                # 🔁 AI's turn in crossfire? Immediately reply
                if state.is_ai_turn():
                    speech_type = state.get_expected_speech_type()
                    print("🤖 AI (crossfire) turn — speech_type:", speech_type)

                    ai_text = await generate_ai_speech(state, speech_type)
                    state.log_speech(
                    speaker="ai",
                    role=state.ai_role,
                    speech_type=speech_type,
                    content=ai_text
                    )

                    await websocket.send_json({
                        "event": "ai_speech",
                        "speech_type": speech_type,
                        "text": ai_text,
                        "state": state.get_state()
                        })

            else:
            # Normal phase transition
                state.current_phase = new_phase
                state.advance_phase()
                print(f"✅ Phase advanced → Now in: {state.current_phase.name}")

                # Send phase update
                await websocket.send_json({
                    "event": "phase_updated",
                    "state": state.get_state()
                })

                # AI Turn for non-crossfire
                if state.is_ai_turn():
                    speech_type = state.get_expected_speech_type()
                    print("🤖 AI turn — speech_type:", speech_type)

                    ai_text = await generate_ai_speech(state, speech_type)
                    state.log_speech(
                    speaker="ai",
                    role=state.ai_role,
                    speech_type=speech_type,
                    content=ai_text
                    )

                    await websocket.send_json({
                    "event": "ai_speech",
                    "speech_type": speech_type,
                    "text": ai_text,
                    "state": state.get_state()
                    })


    except WebSocketDisconnect:
        print("🔌 WebSocket disconnected.")
