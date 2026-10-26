"""Helpers for the XiaoZhi ``llm.emotion`` device expression channel."""

from __future__ import annotations

# Firmware maps "listening" to Expression::Listening (halo + sound-wave arcs).
LISTENING_EMOTION = "listening"
WAKE_CONFIRM_EMOTION = "winking"
IDLE_EMOTION = "neutral"
DEAD_LED_RGB = (255, 0, 0)
# Expression::Dizzy is reserved for "voice output is broken" (red face, x eyes);
# the persona moods never map onto it.
VOICE_ERROR_EMOTION = "dizzy"
VOICE_ERROR_FACE = "voice_error"

SUPPORTED_DEVICE_EMOTIONS = frozenset(
    {
        "neutral",
        "listening",
        "happy",
        "sad",
        "angry",
        "doubt",
        "sleepy",
        "laughing",
        "funny",
        "loving",
        "delicious",
        "kissy",
        "winking",
        "silly",
        "crying",
        "shocked",
        "surprised",
        "thinking",
        "confused",
        "embarrassed",
        "relaxed",
        "cool",
        "confident",
        VOICE_ERROR_EMOTION,
    }
)

FACE_TO_DEVICE_EMOTION = {
    "idle": IDLE_EMOTION,
    "neutral": IDLE_EMOTION,
    "listening": LISTENING_EMOTION,
    "happy": "happy",
    "working_typing": "thinking",
    "juggling": "happy",
    "sweeping": IDLE_EMOTION,
    "sleeping": "sleepy",
    "sleepy": "sleepy",
    "embarrassed": "embarrassed",
    "error": "shocked",
    "notification": "surprised",
    "thinking": "thinking",
    "doubt": "doubt",
    "winking": "winking",
    VOICE_ERROR_FACE: VOICE_ERROR_EMOTION,
}


def normalize_device_emotion(emotion: str | None, *, default: str = IDLE_EMOTION) -> str:
    """Return a firmware-supported emotion name."""
    value = str(emotion or "").strip().lower()
    if value in SUPPORTED_DEVICE_EMOTIONS:
        return value
    return default


def face_to_device_emotion(face: str | None) -> str:
    """Map gateway avatar names onto firmware-supported ``llm.emotion`` names."""
    value = str(face or "").strip().lower()
    return FACE_TO_DEVICE_EMOTION.get(value, normalize_device_emotion(value))
