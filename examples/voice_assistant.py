#!/usr/bin/env python3
"""Simple voice assistant example using PersonaPlex.

Takes a WAV file as input and produces a response WAV and text transcript.
Uses the default assistant prompt: "You are a wise and friendly teacher."

Usage:
    python examples/voice_assistant.py --input assets/test/input_assistant.wav
    python examples/voice_assistant.py --input my_question.wav --output answer.wav --voice NATM0
    python examples/voice_assistant.py --input my_question.wav --device cpu --cpu-offload
"""

import argparse
import os
import sys


def main():
    parser = argparse.ArgumentParser(
        description="PersonaPlex voice assistant — WAV in, response WAV + text out."
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test/input_assistant.wav",
        help="Path to input WAV file (default: assets/test/input_assistant.wav)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_assistant.wav",
        help="Path to output WAV file (default: output_assistant.wav)",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default="output_assistant.json",
        help="Path to output text JSON (default: output_assistant.json)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="NATF2",
        help="Voice name, e.g. NATF2, NATM1, VARF0 (default: NATF2)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: 'cuda' or 'cpu' (default: cuda)",
    )
    parser.add_argument(
        "--cpu-offload",
        action="store_true",
        help="Offload model layers to CPU (requires 'accelerate' package)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42424242,
        help="Random seed for reproducibility (default: 42424242, -1 to disable)",
    )
    args = parser.parse_args()

    # Validate HF_TOKEN before heavy imports
    if not os.environ.get("HF_TOKEN"):
        print(
            "Error: HF_TOKEN environment variable not set.\n"
            "1. Accept the model license: https://huggingface.co/nvidia/personaplex-7b-v1\n"
            "2. Set your token: export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        sys.exit(1)

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    # Heavy imports after validation
    import torch
    from moshi.offline import run_inference, _get_voice_prompt_dir
    from moshi.models import loaders

    # Resolve voice prompt path
    voice_prompt_dir = _get_voice_prompt_dir(None, loaders.DEFAULT_REPO)
    voice_prompt_path = os.path.join(voice_prompt_dir, f"{args.voice}.pt")

    if not os.path.exists(voice_prompt_path):
        print(
            f"Error: Voice prompt not found: {voice_prompt_path}\n"
            f"Available voices: NATF0-3, NATM0-3, VARF0-4, VARM0-4",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print(f"Voice:  {args.voice}")
    print(f"Device: {args.device}")
    print()

    with torch.no_grad():
        run_inference(
            input_wav=args.input,
            output_wav=args.output,
            output_text=args.output_text,
            text_prompt="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
            voice_prompt_path=voice_prompt_path,
            tokenizer_path=None,
            moshi_weight=None,
            mimi_weight=None,
            hf_repo=loaders.DEFAULT_REPO,
            device=args.device,
            seed=args.seed,
            temp_audio=0.8,
            temp_text=0.7,
            topk_audio=250,
            topk_text=25,
            greedy=False,
            save_voice_prompt_embeddings=False,
            cpu_offload=args.cpu_offload,
        )

    print(f"\nDone! Output audio: {args.output}")
    print(f"Output text: {args.output_text}")


if __name__ == "__main__":
    main()
