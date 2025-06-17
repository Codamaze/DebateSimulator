import os
import json
import httpx
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory explicitly
env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(dotenv_path=env_path)

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
MODEL = "deepseek/deepseek-chat:free"

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

    # === SYSTEM PROMPT ===
    greeting_instruction = (
        "Start immediately — do NOT include greetings or meta-commentary. Respond in 1–3 concise sentences."
        if is_crossfire else
        "Begin with a brief formal greeting such as 'Honorable judges, esteemed opponents...' then deliver your argument."
    )

    # Base prompt
    system_prompt = f"""
You are a competitive debater in a Public Forum Debate. Follow the format of the round.

Your assigned role: {side}
Your position: You must strictly **{stance.upper()}** the resolution:
"{debate_state.resolution}"

You are now delivering your **{speech_type}** speech.

Rules:
- NEVER support the other side — even partially.
- Do NOT label your speech as 'Pro Rebuttal' or 'Con Constructive'.
- Do NOT say 'as instructed' or include instructions in your response.

{greeting_instruction}
""".strip()

    # Add structure hints inside appropriate conditionals
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
        system_prompt += "\n" + SPEECH_STRUCTURE_HINTS["Summary"]

    elif is_final_focus:
        system_prompt += "\n" + SPEECH_STRUCTURE_HINTS["Final Focus"]

    else:
        # Default to Constructive
        system_prompt += "\n" + SPEECH_STRUCTURE_HINTS["Constructive"]

    system_prompt += "\n" + difficulty
    messages.append({"role": "system", "content": system_prompt})

    # === CONTEXT STRATEGY ===
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

    # Add only the last user and relevant assistant crossfire context
        for entry in context:
            if entry["speech_type"].lower() == "crossfire":
                role = "assistant" if entry["speaker"] == "ai" else "user"
                messages.append({"role": role, "content": entry["content"]})

    # Force-inject latest user crossfire question
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
        # Constructive, Summary, Final Focus
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

    payload = {
        "model": MODEL,
        "messages": messages,
        "stream": True,
        "temperature": 0.7,
        "repetition_penalty": 1.2,
        "top_p": 0.9
    }

    print("Prompt sent to OpenRouter:")
    print(json.dumps(payload, indent=2))

    ai_response_text = ""

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            async with client.stream("POST", OPENROUTER_URL, json=payload, headers=headers) as response:
                if response.status_code != 200:
                    error_msg = await response.aread()
                    print("OpenRouter API Error:", error_msg.decode())
                    return "LLM error: Unable to generate speech."

                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    line = line.replace("data: ", "").strip()
                    if line == "[DONE]":
                        break
                    try:
                        chunk = json.loads(line)
                        delta = chunk["choices"][0]["delta"]
                        content = delta.get("content", "")
                        ai_response_text += content
                    except Exception as e:
                        print("Failed to parse chunk:", e)
                        print("Raw:", line)
                        continue

    except httpx.RequestError as e:
        print("Network Error:", e)
        return "Network error calling OpenRouter."

    return ai_response_text.strip()
