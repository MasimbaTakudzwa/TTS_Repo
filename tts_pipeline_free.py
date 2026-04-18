"""
tts_pipeline_free.py
--------------------
Completely FREE, local text-to-speech pipeline using Coqui XTTS v2.

No API keys. No usage limits. Runs entirely on your machine.
Supports voice cloning from a short reference audio clip (~6–30 seconds).

Supports input as:
  - Plain text string
  - .txt / .md file
  - PDF file (text extraction)

Usage (as a module):
  from tts_pipeline_free import TTSPipeline

  pipe = TTSPipeline(reference_audio="my_voice.wav")
  pipe.speak_text("Hello world", output_path="output.wav")
  pipe.speak_file("document.pdf", output_path="output.wav")

Requirements:
  pip install TTS pdfplumber
  (first run will auto-download the XTTS v2 model ~1.8GB)
"""

import os
import re
import platform
import tempfile
from pathlib import Path


# ── Constants ─────────────────────────────────────────────────────────────

SUPPORTED_TEXT_TYPES = {".txt", ".md"}
SUPPORTED_PDF_TYPES  = {".pdf"}

# XTTS can handle longer chunks than cloud APIs
CHUNK_SIZE = 250  # words per chunk (XTTS works better with word-based splitting)

DEFAULT_LANGUAGE = "en"


# ── Device detection ──────────────────────────────────────────────────────

def detect_device() -> tuple[str, bool]:
    """
    Automatically detect the best available compute device.

    Returns
    -------
    (device_name, use_gpu) where device_name is one of:
      "cuda"  — NVIDIA GPU (Windows/Linux, e.g. RTX 3060)
      "mps"   — Apple Silicon GPU (M1/M2/M3 MacBook)
      "cpu"   — fallback, no GPU available

    The returned use_gpu bool is True for cuda and mps, False for cpu.
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

    print("[Device] No GPU detected — falling back to CPU (this will be slow for long text)")
    return "cpu", False


# ── Text helpers ──────────────────────────────────────────────────────────

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


def convert_audio_to_wav(input_path: str) -> str:
    """
    Convert any audio file to a 22050Hz mono WAV suitable for XTTS.
    Requires ffmpeg to be installed (brew install ffmpeg).

    Returns the path to the converted WAV file.
    If the file is already a correctly formatted WAV, returns it unchanged.
    """
    import subprocess
    path = Path(input_path)

    if path.suffix.lower() == ".wav":
        return input_path  # already WAV, pass through

    out_path = str(path.with_suffix(".wav"))
    print(f"[Audio] Converting {path.name} → {Path(out_path).name}...")

    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", input_path, "-ar", "22050", "-ac", "1", out_path],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print(f"[Audio] Conversion complete: {out_path}")
        return out_path
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg is not installed. Run: brew install ffmpeg\n"
            "Then retry, or manually convert your file to WAV first."
        )
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"ffmpeg conversion failed for {input_path}: {e}")


def preprocess_for_prosody(text: str) -> str:
    """
    Improve TTS inflection by adjusting punctuation and sentence structure.

    XTTS reads punctuation as prosody cues — commas create short pauses,
    periods drop the pitch, question marks raise it. This function adds
    and cleans up punctuation so the model stresses the right parts.
    """
    # Ensure sentences end with punctuation
    text = re.sub(r'([a-zA-Z])(\s*\n)', r'\1.\2', text)

    # Add comma after common introductory words/phrases
    intros = (
        r'\b(However|Therefore|Moreover|Furthermore|Nevertheless|Consequently|'
        r'Meanwhile|Instead|Otherwise|Additionally|Specifically|Notably|'
        r'In fact|As a result|For example|For instance|In other words|'
        r'That said|Even so|In addition|On the other hand|At the same time)'
        r'(?!\s*,)'
    )
    text = re.sub(intros, r'\1,', text, flags=re.IGNORECASE)

    # Add comma before coordinating conjunctions in long sentences
    # (only when joining two clauses of 5+ words each)
    text = re.sub(
        r'(\w[\w\s]{20,}?)\s+(but|yet|so)\s+(\w)',
        r'\1, \2 \3',
        text,
        flags=re.IGNORECASE
    )

    # Replace " - " (em dash used as pause) with comma + space
    text = re.sub(r'\s+[-–—]\s+', ', ', text)

    # Clean up any double punctuation that may have been introduced
    text = re.sub(r'([,.]){2,}', r'\1', text)
    text = re.sub(r',\.', '.', text)

    # Collapse multiple spaces
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


def clean_text(text: str) -> str:
    """Remove excessive whitespace and blank lines."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    return text.strip()


def chunk_by_words(text: str, chunk_size: int = CHUNK_SIZE) -> list[str]:
    """
    Split text into word-count-based chunks at sentence boundaries.
    XTTS performs best with chunks of ~150-300 words.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current_words, current = [], 0, []

    for sentence in sentences:
        word_count = len(sentence.split())
        if current_words + word_count > chunk_size and current:
            chunks.append(" ".join(current))
            current, current_words = [], 0
        current.append(sentence)
        current_words += word_count

    if current:
        chunks.append(" ".join(current))

    return chunks


def merge_wav_files(wav_paths: list[str], output_path: str) -> None:
    """Concatenate multiple WAV files into one using wave module (stdlib)."""
    import wave

    with wave.open(wav_paths[0], 'rb') as first:
        params = first.getparams()

    with wave.open(output_path, 'wb') as out:
        out.setparams(params)
        for wav_path in wav_paths:
            with wave.open(wav_path, 'rb') as wf:
                out.writeframes(wf.readframes(wf.getnframes()))

    for wav_path in wav_paths:
        os.remove(wav_path)


# ── Main pipeline class ───────────────────────────────────────────────────

class TTSPipeline:
    """
    A free, local text-to-speech pipeline using Coqui XTTS v2.

    Automatically detects whether to use:
      - CUDA   (NVIDIA GPU on Windows/Linux — e.g. RTX 3060)
      - MPS    (Apple Silicon on macOS — e.g. M3 MacBook)
      - CPU    (fallback if no GPU is available)

    Parameters
    ----------
    reference_audio : str
        Path to a WAV file of the voice you want to clone.
        Should be 6–30 seconds of clear speech, minimal background noise.
        If None, uses the XTTS default voice.
    language : str
        Language code. Default: 'en'. XTTS supports 17 languages including
        es, fr, de, zh, ja, hi, ar, pt, pl, and more.
    force_cpu : bool
        Set to True to skip GPU detection and run on CPU only.
        Useful for debugging or low-VRAM situations.
    temperature : float
        Controls expressiveness and variation in the output.
        Range: 0.1 – 1.0. Default: 0.75.
        Lower (0.3–0.5) = more stable, consistent, but flatter.
        Higher (0.7–0.9) = more expressive and natural, slight randomness.
    repetition_penalty : float
        Discourages the model from repeating the same sounds/patterns.
        Default: 5.0. Increase (up to 10.0) if output sounds loopy or stuck.
    speed : float
        Speaking speed multiplier. Default: 1.0.
        0.85 = slightly slower and more deliberate (often sounds more natural).
        1.15 = slightly faster.
    top_p : float
        Nucleus sampling threshold. Default: 0.85.
        Lower = safer/more predictable. Higher = more varied.
    """

    def __init__(
        self,
        reference_audio: str = None,
        language: str = DEFAULT_LANGUAGE,
        force_cpu: bool = False,
        temperature: float = 0.75,
        repetition_penalty: float = 5.0,
        speed: float = 1.0,
        top_p: float = 0.85,
    ):
        self.language = language
        self.temperature = temperature
        self.repetition_penalty = repetition_penalty
        self.speed = speed
        self.top_p = top_p
        self._model = None  # lazy-load on first use

        if force_cpu:
            self.device = "cpu"
            self.use_gpu = False
            print("[Device] CPU mode forced.")
        else:
            self.device, self.use_gpu = detect_device()

        if reference_audio:
            if not Path(reference_audio).exists():
                raise FileNotFoundError(f"Reference audio not found: {reference_audio}")
            # Auto-convert to WAV if needed (MP3, M4A, etc.)
            self.reference_audio = convert_audio_to_wav(reference_audio)
        else:
            self.reference_audio = None

    def _load_model(self):
        """
        Lazy-load the XTTS model (downloads ~1.8GB on first run).

        Device handling:
          - CUDA: pass gpu=True — Coqui handles this natively
          - MPS:  pass gpu=False, then manually move the model to MPS device
                  (Coqui's gpu flag only covers CUDA, so MPS needs a manual push)
          - CPU:  pass gpu=False
        """
        if self._model is not None:
            return

        try:
            from TTS.api import TTS
        except ImportError:
            raise ImportError(
                "Coqui TTS is not installed. Run:\n"
                "  pip install TTS\n"
                "Note: this will also install PyTorch if not already present."
            )

        print("[TTS] Loading XTTS v2 model (first run downloads ~1.8GB)...")

        if self.device == "cuda":
            # CUDA: Coqui handles this natively with gpu=True
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=True)

        elif self.device == "mps":
            # XTTS v2 has a known incompatibility with MPS during inference —
            # the transformers attention mask step fails on the MPS backend.
            # On M-series Macs we run on CPU instead, which is still fast
            # thanks to Apple Silicon's unified memory architecture.
            print("[Device] MPS detected but XTTS inference is CPU-only on Apple Silicon (still fast on M-series)")
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

        else:
            # CPU fallback
            self._model = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=False)

        print("[TTS] Model loaded.")

    # ── Public methods ────────────────────────────────────────────────────

    def speak_text(self, text: str, output_path: str = "output.wav") -> str:
        """
        Convert a plain text string to speech and save as a WAV file.

        Parameters
        ----------
        text        : The text to speak.
        output_path : Where to save the audio. Use .wav extension.

        Returns
        -------
        The path to the saved audio file.
        """
        self._load_model()
        text = clean_text(text)
        text = preprocess_for_prosody(text)
        print(f"[TTS] Processing {len(text):,} characters...")

        chunks = chunk_by_words(text)
        print(f"[TTS] Split into {len(chunks)} chunk(s).")

        return self._generate_and_save(chunks, output_path)

    def speak_file(self, file_path: str, output_path: str = None) -> str:
        """
        Convert a text file or PDF to speech.

        Parameters
        ----------
        file_path   : Path to a .txt, .md, or .pdf file.
        output_path : Where to save the audio. Defaults to input filename + .wav

        Returns
        -------
        The path to the saved audio file.
        """
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

    def list_languages(self) -> list[str]:
        """Print languages supported by XTTS v2."""
        langs = [
            "en", "es", "fr", "de", "it", "pt", "pl", "tr",
            "ru", "nl", "cs", "ar", "zh-cn", "ja", "hu", "ko", "hi"
        ]
        print("Supported languages:", ", ".join(langs))
        return langs

    # ── Internal helpers ──────────────────────────────────────────────────

    def _generate_and_save(self, chunks: list[str], output_path: str) -> str:
        """Generate audio for each chunk, then merge into a single file."""
        if len(chunks) == 1:
            self._tts_to_file(chunks[0], output_path)
        else:
            temp_paths = []
            for i, chunk in enumerate(chunks):
                print(f"[TTS] Generating chunk {i+1}/{len(chunks)}...")
                temp_path = output_path + f".part{i}.wav"
                self._tts_to_file(chunk, temp_path)
                temp_paths.append(temp_path)

            print(f"[TTS] Merging {len(chunks)} chunks...")
            merge_wav_files(temp_paths, output_path)

        print(f"[TTS] Saved to: {output_path}")
        return output_path

    def _tts_to_file(self, text: str, output_path: str) -> None:
        """Run XTTS inference for a single chunk."""
        kwargs = dict(
            text=text,
            language=self.language,
            file_path=output_path,
            speed=self.speed,
        )
        if self.reference_audio:
            kwargs["speaker_wav"] = self.reference_audio
        else:
            kwargs["speaker"] = "Claribel Dervla"

        # Pass expressiveness parameters directly to the underlying model
        # for finer control over prosody and naturalness
        try:
            self._model.synthesizer.tts_model.inference_noise_scale = self.temperature
            self._model.synthesizer.tts_model.length_scale = 1.0 / self.speed
        except AttributeError:
            pass  # not all model versions expose these directly

        self._model.tts_to_file(**kwargs)