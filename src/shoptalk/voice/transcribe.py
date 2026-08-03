"""
Voice input (stretch deliverable, problem statement "Optional Deliverables"):
Whisper speech-to-text, so a user can speak a query instead of typing it.
Runs client-side in the Streamlit process (`faster-whisper`, CPU-friendly)
rather than as an API endpoint -- keeps the API surface focused on the
core search/verify contract, and avoids shipping audio files over HTTP for
a same-machine dev/demo UI.

Usage:
  python -m shoptalk.voice.transcribe --audio path/to/clip.wav
"""
import argparse
import sys

import numpy as np

from shoptalk.config import load_config

_model = None


def _resolve_device(requested: str) -> str:
    if requested != "auto":
        return requested
    import torch

    if torch.cuda.is_available():
        return "cuda"
    return "cpu"  # faster-whisper (CTranslate2) has no MPS backend -- CPU is the Apple Silicon path


def _get_model(cfg: dict = None):
    global _model
    if _model is None:
        from faster_whisper import WhisperModel

        cfg = cfg or load_config()
        vcfg = cfg["voice"]
        device = _resolve_device(vcfg["device"])
        compute_type = "int8" if device == "cpu" else "float16"
        _model = WhisperModel(vcfg["stt_model"], device=device, compute_type=compute_type)
    return _model


def transcribe(audio_path: str, cfg: dict = None) -> str:
    model = _get_model(cfg)
    # macOS Accelerate BLAS emits spurious matmul RuntimeWarnings inside
    # faster-whisper's mel-spectrogram feature extraction (same benign
    # quirk as elsewhere in this project -- verified output is unaffected).
    # `segments` is a lazy generator, so the warning-triggering work
    # happens during iteration, not the transcribe() call itself.
    with np.errstate(all="ignore"):
        segments, _info = model.transcribe(audio_path, beam_size=5)
        text = " ".join(segment.text.strip() for segment in segments).strip()
    return text


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True)
    args = parser.parse_args()

    text = transcribe(args.audio)
    print(f"transcript: {text!r}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
