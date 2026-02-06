#!/usr/bin/env python3
"""Customer service bot example with pre-built persona templates.

Demonstrates how to use PersonaPlex with custom text prompts for different
customer service scenarios. Includes 4 built-in personas taken from the
PersonaPlex prompting guide.

Usage:
    python examples/customer_service_bot.py --list-personas
    python examples/customer_service_bot.py --persona appliance_repair --input assets/test/input_service.wav
    python examples/customer_service_bot.py --custom-prompt "You work for ..." --input call.wav
"""

import argparse
import os
import sys

PERSONAS = {
    "appliance_repair": {
        "name": "SwiftPlex Appliances — Farhod Toshmatov",
        "prompt": (
            "You work for SwiftPlex Appliances which is a appliance repair company "
            "and your name is Farhod Toshmatov. Information: The dishwasher model is "
            "out of stock for replacement parts; we can use an alternative part with "
            "a 3-day delay. Labor cost remains $60 per hour."
        ),
        "voice": "NATM1",
    },
    "restaurant": {
        "name": "Jerusalem Shakshuka — Owen Foster",
        "prompt": (
            "You work for Jerusalem Shakshuka which is a restaurant and your name is "
            "Owen Foster. Information: There are two shakshuka options: Classic (poached "
            "eggs, $9.50) and Spicy (scrambled eggs with jalapenos, $10.25). Sides include "
            "warm pita ($2.50) and Israeli salad ($3). No combo offers. Available for "
            "drive-through until 9 PM."
        ),
        "voice": "NATM0",
    },
    "drone_rental": {
        "name": "AeroRentals Pro — Tomaz Novak",
        "prompt": (
            "You work for AeroRentals Pro which is a drone rental company and your name "
            "is Tomaz Novak. Information: AeroRentals Pro has the following availability: "
            "PhoenixDrone X ($65/4 hours, $110/8 hours), and the premium SpectraDrone 9 "
            "($95/4 hours, $160/8 hours). Deposit required: $150 for standard models, "
            "$300 for premium."
        ),
        "voice": "NATM2",
    },
    "waste_management": {
        "name": "CitySan Services — Ayelen Lucero",
        "prompt": (
            "You work for CitySan Services which is a waste management and your name is "
            "Ayelen Lucero. Information: Verify customer name Omar Torres. Current schedule: "
            "every other week. Upcoming pickup: April 12th. Compost bin service available "
            "for $8/month add-on."
        ),
        "voice": "NATF1",
    },
}


def list_personas():
    """Print available personas in a formatted table."""
    print("Available personas:\n")
    print(f"  {'ID':<20} {'Name':<45} {'Voice'}")
    print(f"  {'-'*20} {'-'*45} {'-'*6}")
    for pid, info in PERSONAS.items():
        print(f"  {pid:<20} {info['name']:<45} {info['voice']}")
    print(f"\nUsage: python examples/customer_service_bot.py --persona <ID> --input <WAV>")


def main():
    parser = argparse.ArgumentParser(
        description="PersonaPlex customer service bot with pre-built personas."
    )
    parser.add_argument(
        "--list-personas",
        action="store_true",
        help="List available personas and exit",
    )
    parser.add_argument(
        "--persona",
        type=str,
        choices=list(PERSONAS.keys()),
        help="Pre-built persona to use",
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Custom text prompt (overrides --persona)",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test/input_service.wav",
        help="Path to input WAV file (default: assets/test/input_service.wav)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output_service.wav",
        help="Path to output WAV file (default: output_service.wav)",
    )
    parser.add_argument(
        "--output-text",
        type=str,
        default="output_service.json",
        help="Path to output text JSON (default: output_service.json)",
    )
    parser.add_argument(
        "--voice",
        type=str,
        help="Override the persona's default voice (e.g. NATF2, NATM1)",
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
        help="Random seed (default: 42424242, -1 to disable)",
    )
    args = parser.parse_args()

    if args.list_personas:
        list_personas()
        sys.exit(0)

    # Determine prompt and voice
    if args.custom_prompt:
        text_prompt = args.custom_prompt
        voice = args.voice or "NATF2"
        print(f"Using custom prompt: {text_prompt[:80]}...")
    elif args.persona:
        persona = PERSONAS[args.persona]
        text_prompt = persona["prompt"]
        voice = args.voice or persona["voice"]
        print(f"Persona: {persona['name']}")
    else:
        print("Error: Specify --persona or --custom-prompt (use --list-personas to see options)", file=sys.stderr)
        sys.exit(1)

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
    voice_prompt_path = os.path.join(voice_prompt_dir, f"{voice}.pt")

    if not os.path.exists(voice_prompt_path):
        print(f"Error: Voice prompt not found: {voice_prompt_path}", file=sys.stderr)
        sys.exit(1)

    print(f"Voice:  {voice}")
    print(f"Input:  {args.input}")
    print(f"Output: {args.output}")
    print()

    with torch.no_grad():
        run_inference(
            input_wav=args.input,
            output_wav=args.output,
            output_text=args.output_text,
            text_prompt=text_prompt,
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
