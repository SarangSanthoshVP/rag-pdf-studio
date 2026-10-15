# Speech-to-Text (STT) and Text-to-Speech (TTS) Module
import os
import io
import re
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

_genai_client = None


def get_genai_client():
    """
    Initialize Google GenAI client for multimodal audio transcription.
    """
    global _genai_client
    if _genai_client is None:
        from google import genai
        key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY or GOOGLE_API_KEY not found in environment.")
        _genai_client = genai.Client(api_key=key)
    return _genai_client


def clean_text_for_speech(text: str) -> str:
    """
    Clean markdown symbols, bullets, URLs, and code blocks for natural speech readout.
    """
    if not text:
        return ""
    # Remove code blocks
    cleaned = re.sub(r"```[\s\S]*?```", " [code snippet omitted] ", text)
    # Remove inline code
    cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)
    # Remove markdown links, keep text
    cleaned = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", cleaned)
    # Remove markdown headers and asterisks
    cleaned = re.sub(r"#{1,6}\s*", "", cleaned)
    cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", cleaned)
    cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)
    cleaned = re.sub(r"_{1,2}([^_]+)_{1,2}", r"\1", cleaned)
    # Replace bullet points with pauses
    cleaned = re.sub(r"^\s*[-*•]\s+", "", cleaned, flags=re.MULTILINE)
    # Collapse multiple whitespace
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def transcribe_audio(audio_bytes: bytes, mime_type: str = "audio/webm", model_name: str = "gemini-3.5-flash-lite") -> str:
    """
    Speech-to-Text (STT):
    Transcribes spoken voice audio into clean text query using Gemini's native multimodal capabilities.
    Zero local RAM overhead (runs in cloud).
    """
    if not audio_bytes:
        raise ValueError("Audio data is empty.")

    client = get_genai_client()
    from google.genai import types

    # Determine suitable mime type
    if not mime_type or mime_type == "application/octet-stream":
        mime_type = "audio/webm"

    prompt = (
        "Transcribe the spoken audio query verbatim into plain text. "
        "Return strictly ONLY the transcribed text question with proper punctuation, without any quotation marks or extra explanation."
    )

    models_to_try = [model_name, "gemini-3.5-flash", "gemini-3.8-flash"]

    last_error = None
    for model in models_to_try:
        try:
            response = client.models.generate_content(
                model=model,
                contents=[
                    types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                    prompt
                ]
            )
            transcription = response.text.strip() if response.text else ""
            # Strip outer quotes if any
            transcription = re.sub(r'^["\']|["\']$', '', transcription).strip()
            if transcription:
                return transcription
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Speech-to-Text transcription failed: {last_error}")


def text_to_speech(text: str, lang: str = "en") -> bytes:
    """
    Text-to-Speech (TTS):
    Converts generated AI response text into spoken audio (MP3 format) using gTTS.
    Runs with near-zero RAM footprint on Render Free Tier.
    """
    cleaned_text = clean_text_for_speech(text)
    if not cleaned_text:
        cleaned_text = "No response content available to read."

    # Limit maximum characters for speech synthesis (around ~1500 chars) to prevent excessive latency
    if len(cleaned_text) > 1500:
        cleaned_text = cleaned_text[:1490] + "... and more."

    try:
        from gtts import gTTS
        tts = gTTS(text=cleaned_text, lang=lang, slow=False)
        buffer = io.BytesIO()
        tts.write_to_fp(buffer)
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        raise RuntimeError(f"Text-to-Speech conversion failed: {e}")
