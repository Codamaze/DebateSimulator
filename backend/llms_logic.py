import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv
import json

# Load .env from backend directory explicitly
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "deepseek/deepseek-chat:free")
FALLBACK_MODEL = os.getenv("FALLBACK_MODEL", "mistralai/mistral-7b-instruct:free")


# --- Prompt Templates ---
DIFFICULTY_PROMPTS = {
    "beginner": "Speak in very simple language. Use only 1–2 core arguments. Avoid complex terminology.",
    "intermediate": "Present clear contentions with warrants and impacts. Focus on logical structure.",
    "expert": "Use advanced weighing, link chains, and preempt opponent logic. Frame for judges strategically. You are a professional-level debater. Use high-level rhetoric, structure, and strategy to dominate the round.",
}

SPEECH_STRUCTURE_HINTS = {
    "Constructive": "Structure: [Introduction → Contentions → Warrant → Impact → Conclusion]",
    "Rebuttal": "Structure: [Summarize opponent → Refute key points → Rebuild own case → Conclusion]",
    "Summary": "Structure: [Summarize main arguments → Highlight key impacts → Emphasize why you win]",
    "Final Focus": "Structure: [Voting issues → Extend winning arguments → Call to decision]",
    "Crossfire": "Structure: [Short question or concise direct answer (1–3 sentences)]"
}

def build_messages(debate_state, speech_type_override=None):
    messages = []

    # Use phase-based speech type unless explicitly overridden
    speech_type = speech_type_override or debate_state.get_expected_speech_type()

    side = 'PROPOSITION' if debate_state.ai_role == 'pro' else 'OPPOSITION'
    stance = 'support' if debate_state.ai_role == 'pro' else 'oppose'
    difficulty = DIFFICULTY_PROMPTS.get(debate_state.ai_difficulty.lower(), DIFFICULTY_PROMPTS["intermediate"])

    is_crossfire = "crossfire" in speech_type.lower()
    is_rebuttal = "rebuttal" in speech_type.lower()
    is_summary = "summary" in speech_type.lower()
    is_final_focus = "final focus" in speech_type.lower()

    # === REFACTORED SYSTEM PROMPT ===
    speech_name = speech_type.title()
    current_turn = debate_state.turn_number
    phase_name = debate_state.current_phase.name.replace("_", " ").title()

    system_prompt = f"""
You are a high-performing AI debater participating in a Public Forum (PF) style debate.

Role: {side}  
Stance: You must **strongly {stance.upper()}** the resolution:  
"{debate_state.resolution}"

Current Speech: **{speech_name}** (Turn {current_turn} in {phase_name} phase)

Objective:
Deliver your speech using formal PF debate tone, persuasive logic, and strategic structure.
Respond *to the judges*, not your opponent. Maintain clarity, professionalism, and credibility.

Rules:
- NEVER support or partially validate the opponent’s case.
- Do NOT say “as an AI” or include meta-commentary.
- Do NOT use phrases like "as instructed" or reference the task.
- Keep tone formal, respectful, and competitive.
- Speak naturally as if addressing a real judging panel.
""".strip()

    # === STRUCTURE HINTS (unchanged) ===
    if is_crossfire:
        system_prompt += """

- ONLY respond to the opponent’s last question or ask one.
- Do not include content from your constructive.
- Do not introduce new arguments.
- Do not include labels or explanations.
Structure: [Short question or concise direct answer (1–3 sentences)]
""".strip()

    elif is_rebuttal:
        system_prompt += """

- Refute opponent’s previous speech directly.
- Do not repeat your Constructive.
- Do not add brand new arguments.
- Rebuild your side’s case and expose flaws in their logic.
Structure: [Summarize opponent → Refute key points → Rebuild own case → Conclusion]
""".strip()

    elif is_summary:
        system_prompt += """
- Address your arguments to the judge, not your opponent.
- Do NOT use 'you' or speak directly to the other side.
- Frame why the judge should vote for your side using impact comparison.
Structure: [Summarize main arguments → Highlight key impacts → Emphasize why you win]
""".strip()

    elif is_final_focus:
        system_prompt += """
- Direct all arguments toward the judge.
- Emphasize key voting issues, not new content.
- Do NOT refer to your opponent as 'you.'
- Explain clearly why your side wins the debate.
Structure: [Voting issues → Extend winning arguments → Call to decision]
""".strip()
    else:
        system_prompt += "\n" + SPEECH_STRUCTURE_HINTS["Constructive"]

    system_prompt += "\n" + difficulty
    messages.append({"role": "system", "content": system_prompt})

    # === CONTEXT STRATEGY (unchanged) ===
    def summarize(text, max_length=150):
        if len(text) > max_length:
            short = text[:max_length]
            last_space = short.rfind(' ')
            return short[:last_space] + "..." if last_space != -1 else short + "..."
        return text

    if is_crossfire:
        context = []
        for entry in reversed(debate_state.transcript):
            if "crossfire" in entry["speech_type"].lower():
                context.insert(0, entry)
                if len(context) >= 4:
                    break
            elif not context and (
                "constructive" in entry["speech_type"].lower() or
                "rebuttal" in entry["speech_type"].lower()
            ):
                if entry["speaker"] == "user":
                    context.insert(0, {
                        "speaker": "user",
                        "content": summarize(entry["content"], max_length=100)
                    })
                break

        for entry in context:
            if entry["speech_type"].lower() == "crossfire":
                role = "assistant" if entry["speaker"] == "ai" else "user"
                messages.append({"role": role, "content": entry["content"]})

        for entry in reversed(context):
            if entry["speaker"] == "user" and "crossfire" in entry["speech_type"].lower():
                messages.append({"role": "user", "content": entry["content"]})
                break

    elif is_rebuttal:
        context = []
        for entry in reversed(debate_state.transcript):
            if len(context) >= 5:
                break
            content = summarize(entry["content"], 200) if "constructive" in entry["speech_type"].lower() else entry["content"]
            context.insert(0, {
                "role": "assistant" if entry["speaker"] == "ai" else "user",
                "content": content
            })
        messages.extend(context)

    else:
        for entry in debate_state.transcript[-6:]:
            role = "assistant" if entry["speaker"] == "ai" else "user"
            messages.append({"role": role, "content": entry["content"]})

    return messages


# --- Main LLM Call Function ---
async def generate_ai_speech(debate_state, speech_type):
    messages = build_messages(debate_state, speech_type)

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload_base = {
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "top_p": 0.9
    }

    async def call_model(model):
        payload = dict(payload_base, model=model)
        ai_response_text = ""
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    raise RuntimeError(await response.aread())
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line = line.replace("data: ", "").strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0]["delta"]
                        ai_response_text += delta.get("content", "")
                    except Exception:
                        continue
        return ai_response_text.strip()

    try:
        return await call_model(PRIMARY_MODEL)
    except Exception as e:
        print("⚠️ Primary model failed:", e)
        try:
            print("🔁 Using fallback model...")
            return await call_model(FALLBACK_MODEL)
        except Exception as f:
            print("❌ Fallback failed:", f)
            return "AI error: Could not generate response."

   
async def generate_judging_feedback(debate_state):
    """
    Uses LLM to judge the full debate transcript.
    Returns a JSON-style evaluation: scores, winner, RFD, and feedback.
    """
    side_map = {"pro": "Proposition", "con": "Opposition"}
    transcript_lines = [
        f"{entry['role']} ({entry['speech_type']}): {entry['content']}"
        for entry in debate_state.transcript
    ]
    transcript_text = "\n".join(transcript_lines)

    messages = [
        {
            "role": "system",
            "content": f"""
You are a debate judge for a Public Forum debate.
The topic is: "{debate_state.resolution}"

Evaluate this round by:
- Comparing arguments and evidence quality
- Assessing refutation and clash
- Identifying persuasive strategies

Return your response strictly in JSON with this format:
{{
  "winner": "Proposition" or "Opposition",
  "score_pro": number (0-100),
  "score_con": number (0-100),
  "rfd": "Reason for decision",
  "feedback_pro": "Constructive feedback for Pro side",
  "feedback_con": "Constructive feedback for Con side"
}}
Only return the JSON — no explanation or commentary outside of it.
""".strip()
        },
        {"role": "user", "content": transcript_text}
    ]

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json"
    }

    payload_base = {
        "messages": messages,
        "stream": False,
        "temperature": 0.3,
    }

    async def call_model(model):
        payload = dict(payload_base, model=model)
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            if response.status_code != 200:
                raise RuntimeError(f"Model error {response.status_code}")
            return json.loads(response.json()["choices"][0]["message"]["content"])

    try:
        return await call_model(PRIMARY_MODEL)
    except Exception as e:
        print("⚠️ Judging with primary model failed:", e)
        try:
            print("🔁 Judging using fallback model...")
            return await call_model(FALLBACK_MODEL)
        except Exception as f:
            print("❌ Judging fallback failed:", f)
            return {"error": "Failed to generate judging feedback."}
    