"""
main.py
-------
Entry point for the Emotion-Aware AI Companion.

Usage:
    python main.py                  # Full mode (camera + voice)
    python main.py --no-camera      # Skip emotion detection
    python main.py --no-voice       # Use text input/output instead of mic/speakers
    python main.py --text-only      # Text-only mode (no camera, no voice)
"""

import argparse
import sys
import speech_recognition as sr

from conversation_manager import ConversationManager


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Emotion-Aware AI Companion — an empathetic chatbot that sees and hears you.",
    )
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable webcam / emotion detection.",
    )
    parser.add_argument(
        "--no-voice",
        action="store_true",
        help="Use keyboard input and text output instead of microphone and speakers.",
    )
    parser.add_argument(
        "--text-only",
        action="store_true",
        help="Text-only mode (equivalent to --no-camera --no-voice).",
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the camera preview window.",
    )
    parser.add_argument(
        "--mic-index",
        type=int,
        help="Index of the microphone to use (run --list-mics to see indices).",
    )
    parser.add_argument(
        "--cam-index",
        type=int,
        default=0,
        help="Index of the camera to use (default 0).",
    )
    parser.add_argument(
        "--list-mics",
        action="store_true",
        help="List available microphone devices and their indices.",
    )
    return parser.parse_args()


def print_banner() -> None:
    """Print a startup banner."""
    banner = r"""
    ╔══════════════════════════════════════════════════════╗
    ║                                                      ║
    ║      🤖  Emotion-Aware AI Companion  🤖              ║
    ║                                                      ║
    ║   I can see your emotions and chat with you.         ║
    ║   Say "goodbye" at any time to end our conversation. ║
    ║   Press Ctrl+C to quit immediately.                  ║
    ║                                                      ║
    ╚══════════════════════════════════════════════════════╝
    """
    print(banner)


def main() -> None:
    """Application entry point."""
    args = parse_args()

    if args.list_mics:
        print("\nAvailable Microphone Devices:")
        print("-" * 30)
        for index, name in enumerate(sr.Microphone.list_microphone_names()):
            print(f"Index {index}: {name}")
        sys.exit(0)

    use_camera = not (args.no_camera or args.text_only)
    use_voice = not (args.no_voice or args.text_only)

    print_banner()

    mode_parts = []
    if use_camera:
        mode_parts.append("Camera (emotion detection)")
    if use_voice:
        mode_parts.append("Voice (mic + speakers)")
    if not use_camera:
        mode_parts.append("No camera")
    if not use_voice:
        mode_parts.append("Text I/O")

    print(f"  Mode: {' | '.join(mode_parts)}\n")

    try:
        manager = ConversationManager(
            use_camera=use_camera,
            use_voice=use_voice,
            show_camera=not args.no_preview,
            mic_index=args.mic_index,
            cam_index=args.cam_index
        )
        manager.start()
    except RuntimeError as exc:
        print(f"\n❌  Fatal error: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n❌  Unexpected error: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
