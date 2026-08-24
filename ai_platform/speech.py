"""Speech AI (spec Part 10) - text-to-speech only.

Uses Windows' built-in SAPI (System.Speech), via PowerShell - a real OS
feature already present, not a downloaded model, and genuinely available
on this machine (verified: 3 installed voices). Speech-to-text is NOT
implemented: the only realistic local option (e.g. openai-whisper) is a
~150MB+ pretrained-model download, and per this project's own rule
against replacing the proprietary pipeline with an external pretrained
model where a from-scratch or OS-native option isn't available, that
tradeoff wasn't taken. TTS uses the OS's own synthesis engine, not a
downloaded neural model, so it doesn't raise the same concern.
"""

import os
import subprocess
import tempfile

_PS_SCRIPT = r"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
{voice_line}
$synth.SetOutputToWaveFile('{out_path}')
$synth.Speak([Console]::In.ReadToEnd())
$synth.Dispose()
"""


class SpeechError(RuntimeError):
    pass


def list_voices() -> list:
    result = subprocess.run(
        ["powershell.exe", "-NoProfile", "-Command",
         "Add-Type -AssemblyName System.Speech; "
         "(New-Object System.Speech.Synthesis.SpeechSynthesizer)."
         "GetInstalledVoices() | ForEach-Object { $_.VoiceInfo.Name }"],
        capture_output=True, text=True, timeout=15,
    )
    if result.returncode != 0:
        raise SpeechError(f"Could not list voices: {result.stderr}")
    return [v.strip() for v in result.stdout.splitlines() if v.strip()]


def synthesize(text: str, out_path: str = None, voice: str = None, timeout: float = 20.0) -> str:
    """Real text-to-speech via Windows SAPI. Returns the path to a WAV
    file containing actual synthesized audio. Raises SpeechError on any
    failure - never returns a path to a file that wasn't really produced."""
    if not text or not text.strip():
        raise SpeechError("Cannot synthesize empty text")

    if out_path is None:
        out_path = tempfile.NamedTemporaryFile(suffix=".wav", delete=False).name

    voice_line = ""
    if voice:
        voice_line = f"$synth.SelectVoice('{voice}')"

    script = _PS_SCRIPT.format(voice_line=voice_line, out_path=out_path.replace("\\", "\\\\"))

    try:
        result = subprocess.run(
            ["powershell.exe", "-NoProfile", "-Command", script],
            input=text, capture_output=True, text=True, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise SpeechError(f"Synthesis exceeded {timeout}s timeout")

    if result.returncode != 0 or not os.path.exists(out_path):
        raise SpeechError(f"Synthesis failed: {result.stderr.strip()[-300:]}")

    size = os.path.getsize(out_path)
    if size < 100:  # a real WAV header alone is 44 bytes; near-empty means no real audio
        raise SpeechError(f"Output file is suspiciously small ({size} bytes) - synthesis likely failed silently")

    return out_path
