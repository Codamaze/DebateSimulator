# local_audio/tts_utils.py

import pyttsx3

def speak_aloud(text: str):
    """
    Speak the given text aloud using Hazel's voice (if available).
    """
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)

    # Try to use Hazel voice if present
    voices = engine.getProperty("voices")
    hazel_voice_found = False
    for voice in voices:
        if "Hazel" in voice.name:
            engine.setProperty("voice", voice.id)
            hazel_voice_found = True
            break

    if not hazel_voice_found:
        print("⚠️ 'Hazel' voice not found. Using default voice.")

    engine.say(text)
    engine.runAndWait()
