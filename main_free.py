"""
main_free.py
------------
CLI entry point for the FREE local TTS pipeline (Coqui XTTS v2).

Usage examples:
  # Clone a voice and speak a string
  python main_free.py --text "Hello, this is my cloned voice." --ref my_voice.wav

  # Speak from a file using a cloned voice
  python main_free.py --file notes.txt --ref my_voice.wav

  # Speak a PDF (no voice cloning — uses built-in voice)
  python main_free.py --file document.pdf

  # Use a different language
  python main_free.py --text "Bonjour le monde" --ref ref.wav --lang fr

  # List supported languages
  python main_free.py --list-langs
"""

import argparse
import sys
from tts_pipeline_free import TTSPipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Free local TTS pipeline using Coqui XTTS v2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument("--text", "-t", type=str,
                             help="Text string to speak.")
    input_group.add_argument("--file", "-f", type=str,
                             help="Path to a .txt, .md, or .pdf file.")

    parser.add_argument("--ref", "-r", type=str, default=None,
                        help="Path to reference WAV audio for voice cloning (6–30 sec).")
    parser.add_argument("--output", "-o", type=str, default=None,
                        help="Output WAV file path.")
    parser.add_argument("--lang", "-l", type=str, default="en",
                        help="Language code (default: en). Run --list-langs to see options.")
    parser.add_argument("--cpu", action="store_true",
                        help="Force CPU mode, skipping GPU detection. Useful for debugging.")
    parser.add_argument("--list-langs", action="store_true",
                        help="List supported languages and exit.")

    return parser.parse_args()


def main():
    args = parse_args()

    if args.list_langs:
        TTSPipeline().list_languages()
        sys.exit(0)

    if not args.text and not args.file:
        print("[Error] Provide either --text or --file.", file=sys.stderr)
        sys.exit(1)

    try:
        pipeline = TTSPipeline(
            reference_audio=args.ref,
            language=args.lang,
            force_cpu=args.cpu,
        )
    except FileNotFoundError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)

    try:
        if args.text:
            output = pipeline.speak_text(args.text, output_path=args.output or "output.wav")
        else:
            output = pipeline.speak_file(args.file, output_path=args.output)

        print(f"\n✓ Audio saved to: {output}")

    except Exception as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()