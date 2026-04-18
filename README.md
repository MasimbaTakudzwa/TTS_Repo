# TTS Pipeline — Text to Voice with Voice Cloning

A Python pipeline that converts text, `.txt`/`.md` files, and PDFs into speech.
Includes two versions: a **free local** version (recommended) and a paid API version.

---

## Which version should I use?

| | Free (Coqui XTTS) | Paid (ElevenLabs) |
|---|---|---|
| Cost | Free forever | ~$5–$22/month |
| Internet required | No | Yes |
| Voice cloning | Yes (from WAV file) | Yes (from dashboard) |
| Quality | Very good | Excellent |
| Speed | Slower on CPU, fast on GPU | Fast |
| Setup | Install + download model (~1.8GB) | API key only |

**Recommendation: start with the free version.** You can always switch later.

---

## Setup

### 1. Install dependencies

```bash
pip install TTS pdfplumber
```

> First run will automatically download the XTTS v2 model (~1.8 GB). This only happens once.

### 2. Prepare a reference audio file (for voice cloning)

Record or find a clean WAV file of the voice you want to clone.

- Length: **6–30 seconds** is ideal
- Format: WAV (16kHz or 22kHz mono works best)
- Quality: minimal background noise, clear speech

To convert an MP3 to WAV:
```bash
ffmpeg -i my_recording.mp3 -ar 22050 -ac 1 my_voice.wav
```

---

## Usage

### From the command line

```bash
# Clone a voice and speak a text string
python main_free.py --text "Hello, this is my cloned voice." --ref my_voice.wav

# Speak from a text file
python main_free.py --file notes.txt --ref my_voice.wav --output notes_audio.wav

# Speak from a PDF
python main_free.py --file document.pdf --ref my_voice.wav

# Use without voice cloning (built-in voice)
python main_free.py --text "Hello world"

# Use a different language
python main_free.py --text "Bonjour le monde" --ref ref.wav --lang fr

# Run on CPU only (if you don't have a GPU)
python main_free.py --text "Hello" --ref my_voice.wav --no-gpu

# List supported languages
python main_free.py --list-langs
```

### As a Python module

```python
from tts_pipeline_free import TTSPipeline

# With voice cloning
pipe = TTSPipeline(reference_audio="my_voice.wav")
pipe.speak_text("Hello, this is my cloned voice.", output_path="hello.wav")

# From a file
pipe.speak_file("my_notes.txt", output_path="notes.wav")
pipe.speak_file("document.pdf", output_path="doc_audio.wav")

# No voice cloning (uses built-in XTTS voice)
pipe = TTSPipeline()
pipe.speak_text("Hello world")
```

---

## Supported Languages

XTTS v2 supports 17 languages:

`en` `es` `fr` `de` `it` `pt` `pl` `tr` `ru` `nl` `cs` `ar` `zh-cn` `ja` `hu` `ko` `hi`

---

## Project Structure

```
tts_pipeline/
├── tts_pipeline_free.py   ← Core pipeline (Coqui XTTS — FREE)
├── main_free.py           ← CLI for free pipeline
├── tts_pipeline.py        ← Core pipeline (ElevenLabs — paid)
├── main.py                ← CLI for paid pipeline
├── requirements.txt       ← Dependencies
└── README.md              ← This file
```

---

## Tips

- **GPU strongly recommended.** On CPU, generating a minute of audio can take several minutes. On a mid-range GPU it takes seconds.
- **Longer reference audio = better cloning.** 15–20 seconds of clean speech gives noticeably better results than 6 seconds.
- **Chunk size is tunable.** If output sounds rushed or unnatural, reduce `CHUNK_SIZE` in `tts_pipeline_free.py`.
- **Output is WAV.** Use `ffmpeg` to convert to MP3 if needed: `ffmpeg -i output.wav output.mp3`