"""
tts_pipeline_free.py
--------------------
Enhanced local text-to-speech pipeline using Coqui XTTS v2 (idiap fork).

Optimizations over the base version:
  - Reference audio is resampled to XTTS's native 24kHz, silence-trimmed,
    and loudness-normalized for consistent cloning quality.
  - Accepts a list of reference clips to build a richer speaker embedding.
  - Actual XTTS inference parameters (temperature, repetition_penalty,
    top_k, top_p, length_penalty) are passed through correctly — the
    previous version was setting VITS parameters that XTTS ignores.
  - Chunk size tuned to XTTS's sweet spot (~180 words).
  - Chunks are crossfaded on merge to eliminate audible seams.
  - Text preprocessing expands abbreviations, spells out long numbers,
    strips URLs/emails, and cleans punctuation for better prosody.
  - Final output is peak-normalized.

No API keys. No usage limits. Runs entirely on your machine.
Supports voice cloning from short reference audio (6–30 seconds each).

Usage (as a module):
  from tts_pipeline_free import TTSPipeline

  pipe = TTSPipeline(reference_audio="my_voice.mp3")
  pipe.speak_text("Hello world", output_path="output.wav")

  # Multiple references for better cloning:
  pipe = TTSPipeline(reference_audio=["calm.wav", "excited.wav", "neutral.wav"])

Requirements:
  pip install coqui-tts pdfplumber
  ffmpeg installed and on PATH
  (first run will auto-download the XTTS v2 model ~1.8GB)
"""

import os
import re
import platform
import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional, Union


# ── Constants ─────────────────────────────────────────────────────────────

SUPPORTED_TEXT_TYPES = {".txt", ".md"}
SUPPORTED_PDF_TYPES = {".pdf"}

# XTTS v2 operates natively at 24kHz
XTTS_SAMPLE_RATE = 24000

# XTTS's sweet spot is ~150-200 words per chunk. Beyond ~400 tokens
# (roughly 200 words) the model loses coherence and produces artifacts.
CHUNK_SIZE = 180

DEFAULT_LANGUAGE = "en"


# ── Device detection ──────────────────────────────────────────────────────

def detect_device() -> tuple[str, bool]:
    """
    Pick the best available compute device.
    Returns (device_name, use_gpu) where device_name ∈ {cuda, mps, cpu}.
    """
    try:
        import torch
    except ImportError:
        return "cpu", False

    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"[Device] NVIDIA GPU detected: {gpu_name} — using CUDA")
        return "cuda", True

    if platform.system() == "Darwin" and torch.backends.mps.is_available():
        chip = platform.processor() or "Apple Silicon"
        print(f"[Device] Apple Silicon detected ({chip}) — using MPS")
        return "mps", True

    print("[Device] No GPU detected — falling back to CPU (slow for long text)")
    return "cpu", False


# ── File readers ──────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> str:
    """Extract plain text from a PDF using pdfplumber."""
    try:
        import pdfplumber
    except ImportError:
        raise ImportError("Run: pip install pdfplumber")

    parts = []
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            text = page.extract_text()
            if text:
                parts.append(text.strip())
    return "\n\n".join(parts)


def read_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


# ── Reference audio preprocessing ─────────────────────────────────────────

def _check_ffmpeg() -> None:
    """Raise a helpful error if ffmpeg isn't on PATH."""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        raise RuntimeError(
            "ffmpeg is required but not found on PATH. Install it:\n"
            '  Windows: winget install "FFmpeg (Essentials Build)"\n'
            "  macOS:   brew install ffmpeg\n"
            "  Linux:   sudo apt install ffmpeg\n"
            "Then restart your terminal."
        )


def preprocess_reference_audio(
    input_path: str,
    output_path: Optional[str] = None,
    target_sr: int = XTTS_SAMPLE_RATE,
    trim_silence: bool = True,
    normalize_loudness: bool = True,
) -> str:
    """
    Prepare reference audio for optimal XTTS voice cloning.

    Pipeline (all via ffmpeg):
      1. Convert to mono
      2. Resample to XTTS's native 24kHz (no upsampling inside the model)
      3. Trim leading and trailing silence
      4. EBU R128 loudness normalization to -16 LUFS (broadcast standard)

    A good reference clip:
      - 6-30 seconds of clean single-speaker speech
      - Matches the register you want the clone to produce
      - No background music, noise, or reverb
      - Slight natural variation in pitch (not monotone)
    """
    _check_ffmpeg()
    input_path = str(Path(input_path).resolve())

    if output_path is None:
        # Write alongside the original with a clear suffix, so re-runs skip work
        p = Path(input_path)
        output_path = str(p.with_name(f"{p.stem}_xtts_ref.wav"))

    # Build filter chain
    filters = []
    if trim_silence:
        # Trim up to 10 seconds of silence below -50dB at both ends,
        # but preserve in-speech pauses.
        filters.append(
            "silenceremove=start_periods=1:start_duration=0.1:start_threshold=-50dB:"
            "stop_periods=1:stop_duration=0.3:stop_threshold=-50dB:detection=peak"
        )
    if normalize_loudness:
        # -16 LUFS is the YouTube/podcast standard; gives the clone consistent
        # perceived volume regardless of how the source was recorded.
        filters.append("loudnorm=I=-16:TP=-1.5:LRA=11")

    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", str(target_sr), "-ac", "1"]
    if filters:
        cmd.extend(["-af", ",".join(filters)])
    cmd.append(output_path)

    print(f"[Audio] Preprocessing reference: {Path(input_path).name} → {Path(output_path).name}")

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    except subprocess.CalledProcessError:
        # Filters can fail on very short clips. Fall back to plain conversion.
        print("[Audio] Filter chain failed — falling back to plain conversion")
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", str(target_sr), "-ac", "1", output_path],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        )

    return output_path


# ── Text preprocessing ────────────────────────────────────────────────────

# Abbreviations XTTS mispronounces if read literally
_ABBREVIATIONS = [
    (r'\bMr\.', 'Mister'),
    (r'\bMrs\.', 'Misses'),
    (r'\bMs\.', 'Miss'),
    (r'\bDr\.', 'Doctor'),
    (r'\bSt\.', 'Saint'),
    (r'\bFt\.', 'Fort'),
    (r'\bJr\.', 'Junior'),
    (r'\bSr\.', 'Senior'),
    (r'\bProf\.', 'Professor'),
    (r'\be\.g\.', 'for example,'),
    (r'\bi\.e\.', 'that is,'),
    (r'\betc\.', 'etcetera'),
    (r'\bvs\.', 'versus'),
    (r'\bNo\.(?=\s*\d)', 'Number'),
    (r'\bapprox\.', 'approximately'),
    (r'\bJan\.', 'January'),
    (r'\bFeb\.', 'February'),
    (r'\bAug\.', 'August'),
    (r'\bSept?\.', 'September'),
    (r'\bOct\.', 'October'),
    (r'\bNov\.', 'November'),
    (r'\bDec\.', 'December'),
]


def preprocess_text(text: str) -> str:
    """
    Normalize and prepare text for natural XTTS synthesis.

    Handles the common things that sound obviously synthetic:
      - Abbreviations (Dr., Mr., etc.)
      - Large numbers (spelled out for cleaner pronunciation)
      - URLs and email addresses (stripped — XTTS tries to spell them)
      - Em/en dashes used as pauses (converted to commas)
      - Stray double punctuation and whitespace
    """
    # Whitespace
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text).strip()

    # Kill URLs and emails before anything else
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub(r'www\.\S+', '', text)
    text = re.sub(r'\S+@\S+\.\S+', '', text)

    # Expand abbreviations
    for pattern, replacement in _ABBREVIATIONS:
        text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

    # Spell out numbers of 4+ digits (shorter numbers XTTS handles fine)
    try:
        import inflect
        engine = inflect.engine()

        def _num_to_words(match: re.Match) -> str:
            raw = match.group(0).replace(',', '')
            try:
                return engine.number_to_words(int(raw), andword='')
            except Exception:
                return match.group(0)

        text = re.sub(r'\b\d{1,3}(?:,\d{3})+\b', _num_to_words, text)  # 1,000,000
        text = re.sub(r'\b\d{4,}\b', _num_to_words, text)              # 1000, 2025
    except ImportError:
        pass

    # Ensure line-ending letters have terminal punctuation (prevents run-ons)
    text = re.sub(r'([A-Za-z0-9])(\s*\n)', r'\1.\2', text)

    # Em-dash / en-dash as pause → comma
    text = re.sub(r'\s*[—–]\s*', ', ', text)
    # Hyphen as pause (with spaces around it) → comma
    text = re.sub(r'(\w)\s+-\s+(\w)', r'\1, \2', text)

    # Cleanup: runs of punctuation, comma-period sequences, extra spaces
    text = re.sub(r'([,.!?]){2,}', r'\1', text)
    text = re.sub(r',\s*\.', '.', text)
    text = re.sub(r'\.\s*,', '.', text)
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def chunk_by_words(text: str, chunk_size: int = CHUNK_SIZE) -> List[str]:
    """
    Split text into chunks at sentence boundaries, preferring paragraph breaks.
    Targets ~chunk_size words per chunk (XTTS sweet spot ≈ 180).
    """
    paragraphs = text.split('\n\n')
    chunks: List[str] = []
    current: List[str] = []
    current_words = 0

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        sentences = re.split(r'(?<=[.!?])\s+', para)
        for sentence in sentences:
            word_count = len(sentence.split())
            # If adding this sentence would overflow AND we already have content, flush.
            if current_words + word_count > chunk_size and current:
                chunks.append(" ".join(current))
                current, current_words = [], 0
            current.append(sentence)
            current_words += word_count
        # At a paragraph boundary, flush if the current chunk is already substantial.
        if current_words >= chunk_size * 0.6:
            chunks.append(" ".join(current))
            current, current_words = [], 0

    if current:
        chunks.append(" ".join(current))

    return chunks


# ── WAV handling ──────────────────────────────────────────────────────────

def merge_wav_files(
    wav_paths: List[str],
    output_path: str,
    crossfade_ms: int = 30,
    peak_normalize_db: float = -1.0,
) -> None:
    """
    Concatenate WAV files with a short crossfade to smooth chunk seams,
    then peak-normalize the final output.
    """
    import numpy as np
    import soundfile as sf

    segments = []
    sr = None
    for path in wav_paths:
        data, rate = sf.read(path)
        if data.ndim > 1:
            data = data.mean(axis=1)  # downmix to mono defensively
        if sr is None:
            sr = rate
        segments.append(data.astype(np.float32))

    xf = max(1, int(crossfade_ms * sr / 1000))
    fade_out = np.linspace(1.0, 0.0, xf, dtype=np.float32)
    fade_in = np.linspace(0.0, 1.0, xf, dtype=np.float32)

    merged = segments[0].copy()
    for seg in segments[1:]:
        if len(merged) > xf and len(seg) > xf:
            tail = merged[-xf:] * fade_out
            head = seg[:xf] * fade_in
            merged = np.concatenate([merged[:-xf], tail + head, seg[xf:]])
        else:
            merged = np.concatenate([merged, seg])

    # Peak-normalize to just under 0dBFS to avoid clipping
    peak = np.abs(merged).max()
    if peak > 0:
        target = 10 ** (peak_normalize_db / 20.0)
        merged = merged * (target / peak)

    sf.write(output_path, merged, sr)

    # Clean up partial files
    for path in wav_paths:
        try:
            os.remove(path)
        except OSError:
            pass


# ── Main pipeline class ───────────────────────────────────────────────────

class TTSPipeline:
    """
    Local voice-cloning TTS pipeline powered by XTTS v2.

    Device is auto-detected: CUDA (NVIDIA) → MPS (Apple) → CPU.

    Parameters
    ----------
    reference_audio : str | Path | list of str/Path | None
        One or more reference clips of the target voice (6–30s each).
        Providing 2-4 varied clips usually produces a better clone than
        one long clip. Any format ffmpeg can read. None → default voice.
    language : str
        XTTS language code. Default: 'en'. See list_languages().
    force_cpu : bool
        Skip GPU detection.
    temperature : float (0.1–1.0, default 0.75)
        Expressiveness. Lower = flatter/stabler. Higher = more varied but
        risks occasional weird pronunciations.
    repetition_penalty : float (default 5.0)
        Discourages the model from looping. Raise to 7-10 if the output
        sounds stuck, gets stammery, or repeats phonemes.
    top_k : int (default 50)
        Sampling diversity. 30-70 is a sensible range.
    top_p : float (default 0.85)
        Nucleus sampling. Lower = safer choices, higher = more variety.
    length_penalty : float (default 1.0)
        >1 favors longer outputs, <1 favors shorter.
    speed : float (default 1.0)
        Playback speed. 0.9 often sounds more natural/deliberate than 1.0.
    preprocess_reference : bool (default True)
        Apply silence trimming and loudness normalization to reference audio.
    preprocess_text_input : bool (default True)
        Apply abbreviation/number/URL cleanup to input text.
    """

    def __init__(
        self,
        reference_audio: Union[str, Path, List[Union[str, Path]], None] = None,
        language: str = DEFAULT_LANGUAGE,
        force_cpu: bool = False,
        temperature: float = 0.55,
        repetition_penalty: float = 5.0,
        top_k: int = 50,
        top_p: float = 0.85,
        length_penalty: float = 1.0,
        speed: float = 1.3,
        preprocess_reference: bool = True,
        preprocess_text_input: bool = True,
    ):
        self.language = language
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p
        self.length_penalty = length_penalty
        self.speed = speed
        self.preprocess_text_input = preprocess_text_input
        self._model = None

        if force_cpu:
            self.device, self.use_gpu = "cpu", False
            print("[Device] CPU mode forced.")
        else:
            self.device, self.use_gpu = detect_device()

        self.reference_audio = self._resolve_reference(reference_audio, preprocess_reference)

    # ── Reference resolution ──────────────────────────────────────────────

    def _resolve_reference(
        self,
        reference_audio: Union[str, Path, List, None],
        preprocess: bool,
    ) -> Union[str, List[str], None]:
        if reference_audio is None:
            return None

        # Normalize to list
        if isinstance(reference_audio, (str, Path)):
            refs = [str(reference_audio)]
        else:
            refs = [str(r) for r in reference_audio]

        processed = []
        for ref in refs:
            if not Path(ref).exists():
                raise FileNotFoundError(f"Reference audio not found: {ref}")
            if preprocess:
                processed.append(preprocess_reference_audio(ref))
            else:
                # Still need a WAV for XTTS
                _check_ffmpeg()
                p = Path(ref)
                if p.suffix.lower() == ".wav":
                    processed.append(ref)
                else:
                    out = str(p.with_suffix(".wav"))
                    subprocess.run(
                        ["ffmpeg", "-y", "-i", ref, "-ar", str(XTTS_SAMPLE_RATE), "-ac", "1", out],
                        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                    )
                    processed.append(out)

        # XTTS accepts either a single path or a list
        return processed if len(processed) > 1 else processed[0]

    # ── Model loading ─────────────────────────────────────────────────────

    def _load_model(self):
        if self._model is not None:
            return

        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "coqui-tts is not installed. Run:\n"
                "  pip install coqui-tts\n"
                "Note: install PyTorch separately first."
            )

        print("[TTS] Loading XTTS v2 (first run downloads ~1.8GB)...")

        if self.device == "cuda":
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)
        else:
            # XTTS inference on MPS is broken in practice — stay on CPU on Mac
            if self.device == "mps":
                print("[Device] Using CPU for XTTS inference (MPS has known XTTS issues)")
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

        print("[TTS] Model loaded.")

    # ── Public API ────────────────────────────────────────────────────────

    def speak_text(self, text: str, output_path: str = "output.wav") -> str:
        """Synthesize a string to a WAV file."""
        self._load_model()

        if self.preprocess_text_input:
            text = preprocess_text(text)
        print(f"[TTS] Processing {len(text):,} characters...")

        chunks = chunk_by_words(text)
        print(f"[TTS] Split into {len(chunks)} chunk(s) (target ~{CHUNK_SIZE} words each).")

        return self._generate_and_save(chunks, output_path)

    def speak_file(self, file_path: str, output_path: Optional[str] = None) -> str:
        """Synthesize a .txt / .md / .pdf file to a WAV file."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")

        suffix = path.suffix.lower()
        if suffix in SUPPORTED_PDF_TYPES:
            print(f"[TTS] Extracting text from PDF: {path.name}")
            text = extract_text_from_pdf(str(path))
        elif suffix in SUPPORTED_TEXT_TYPES:
            print(f"[TTS] Reading text file: {path.name}")
            text = read_text_file(str(path))
        else:
            raise ValueError(
                f"Unsupported file type '{suffix}'. "
                f"Supported: {SUPPORTED_TEXT_TYPES | SUPPORTED_PDF_TYPES}"
            )

        if not output_path:
            output_path = str(path.with_suffix(".wav"))
        return self.speak_text(text, output_path)

    def list_languages(self) -> List[str]:
        langs = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi",
        ]
        print("Supported languages:", ", ".join(langs))
        return langs

    # ── Internals ─────────────────────────────────────────────────────────

    def _generate_and_save(self, chunks: List[str], output_path: str) -> str:
        if len(chunks) == 1:
            self._tts_to_file(chunks[0], output_path)
        else:
            temp_paths = []
            for i, chunk in enumerate(chunks):
                print(f"[TTS] Generating chunk {i + 1}/{len(chunks)}...")
                temp_paths.append(self._tts_to_file(chunk, f"{output_path}.part{i}.wav"))
            print(f"[TTS] Merging {len(chunks)} chunks with crossfade...")
            merge_wav_files(temp_paths, output_path)

        print(f"[TTS] Saved to: {output_path}")
        return output_path

    def _tts_to_file(self, text: str, output_path: str) -> str:
        """Run XTTS inference for a single chunk with full parameter control."""
        kwargs = dict(
            text=text,
            language=self.language,
            file_path=output_path,
            # These flow through to XTTS's inference() method.
            # In the previous version these were set as VITS-style attributes
            # (inference_noise_scale, length_scale) which XTTS silently ignores.
            temperature=self.temperature,
            repetition_penalty=self.repetition_penalty,
            top_k=self.top_k,
            top_p=self.top_p,
            length_penalty=self.length_penalty,
            speed=self.speed,
            # We've already chunked intelligently; don't let Coqui re-split.
            split_sentences=False,
        )

        if self.reference_audio:
            kwargs["speaker_wav"] = self.reference_audio
        else:
            kwargs["speaker"] = "Claribel Dervla"

        self._model.tts_to_file(**kwargs)
        return output_path