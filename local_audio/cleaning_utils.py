import re

def clean_speech(text: str) -> str:
    """
    Clean AI-generated text to make it sound better in TTS (Hazel).
    """
    # Remove markdown/formatting artifacts
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)  # bold
    text = re.sub(r"\*(.*?)\*", r"\1", text)      # italic
    text = re.sub(r"\[.*?\]\(.*?\)", "", text)    # links

    # Remove markdown headers and bullets
    text = re.sub(r"#+\s*", "", text)             # headings
    text = re.sub(r"[-*•]\s*", "", text)          # list bullets

    # Replace newline with space
    text = text.replace("\n", " ")

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text
