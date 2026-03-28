"""Audio helpers and transcription utilities."""

import numpy as np
import sounddevice as sd

# ── Globals ──────────────────────────────────────────────────────────
# Lightweight model — used for idle wake-word scanning (fast, low CPU)
WAKE_MODEL   = "mlx-community/whisper-base-mlx"
# High-accuracy model — used for active listening after wake
ACTIVE_MODEL = "mlx-community/whisper-large-v3-mlx"
SAMPLE_RATE  = 16000



def log(tag: str, msg: str = ""):
    """Formatted log: [TAG] message"""
    print(f"[{tag}] {msg}", flush=True)


# ── Audio ────────────────────────────────────────────────────────────
def record_chunk(duration: float = 7.0) -> np.ndarray:
    """Record a fixed-length audio chunk (16 kHz, mono, float32)."""
    log("RECORDING", f"{duration}s — speak now…")
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


def record_short(duration: float = 2.0) -> np.ndarray:
    """Record a short chunk for wake-word scanning."""
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
    )
    sd.wait()
    return audio.flatten()


# ── Whisper (mlx-whisper — Apple Silicon native) ─────────────────────
# Resolved local cache paths — populated once by load_whisper_model()
_wake_path:   str | None = None
_active_path: str | None = None


def load_whisper_model():
    """Resolve both models to local cache paths (downloads on first run).
    All subsequent transcribe() calls use the local path — no HF network calls.
    """
    global _wake_path, _active_path
    from huggingface_hub import snapshot_download
    import mlx_whisper

    log("MODEL", f"Resolving wake model  : {WAKE_MODEL}…")
    _wake_path = snapshot_download(repo_id=WAKE_MODEL)
    log("MODEL", f"Wake model ready  → {_wake_path}")

    log("MODEL", f"Resolving active model: {ACTIVE_MODEL} (first run ~1.5 GB)…")
    _active_path = snapshot_download(repo_id=ACTIVE_MODEL)
    log("MODEL", f"Active model ready → {_active_path}")

    # Warm up both so first real inference isn't slow
    silence = np.zeros(SAMPLE_RATE, dtype=np.float32)
    mlx_whisper.transcribe(silence, path_or_hf_repo=_wake_path)
    mlx_whisper.transcribe(silence, path_or_hf_repo=_active_path)
    log("MODEL", "Both models warmed up.")


def transcribe(audio: np.ndarray, model: str = "wake") -> str:
    """Transcribe audio array → string using mlx-whisper (local path, no network).

    Args:
        audio: 16 kHz float32 numpy array
        model: "wake" (default) or "active"
    """
    import mlx_whisper
    path = _active_path if model == "active" else _wake_path
    if path is None:
        raise RuntimeError("Call load_whisper_model() before transcribe().")
    result = mlx_whisper.transcribe(audio, path_or_hf_repo=path, language="en")
    return result.get("text", "").strip()



# ── TTS — Kokoro ONNX (human male voice) ────────────────────────────
_kokoro = None  # lazy-loaded on first speak()

def _load_kokoro():
    """Download & cache Kokoro model on first call."""
    global _kokoro
    if _kokoro is not None:
        return _kokoro
    from kokoro_onnx import Kokoro
    from huggingface_hub import hf_hub_download
    log("TTS", "Loading Kokoro model (first run: ~350 MB download)…")
    onnx  = hf_hub_download("hexgrad/Kokoro-82M", "kokoro-v0_19.onnx")
    voices = hf_hub_download("hexgrad/Kokoro-82M", "voices.bin")
    _kokoro = Kokoro(onnx, voices)
    log("TTS", "Kokoro ready.")
    return _kokoro


def speak(text: str):
    """Speak text using Kokoro (human male voice). Falls back to macOS say."""
    try:
        k = _load_kokoro()
        samples, sr = k.create(text, voice="am_michael", speed=1.0, lang="en-us")
        sd.play(samples, sr)
        sd.wait()
    except Exception as e:
        log("TTS:WARN", f"Kokoro failed ({e}), falling back to say…")
        try:
            import subprocess
            subprocess.run(["say", "-v", "Daniel", text], check=False)
        except Exception:
            pass  # TTS is optional; never crash the agent
