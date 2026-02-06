#!/usr/bin/env python3
"""Compare all 18 PersonaPlex voices on the same input audio.

Loads the model once, then generates a response for each selected voice.
Useful for auditioning voices or creating comparison demos.

Usage:
    python examples/custom_voice.py --list-voices
    python examples/custom_voice.py --input assets/test/input_assistant.wav
    python examples/custom_voice.py --input my_audio.wav --voices NATF2 NATM1 VARF0
    python examples/custom_voice.py --input my_audio.wav --output-dir voice_samples/
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

ALL_VOICES = {
    "Natural Female": ["NATF0", "NATF1", "NATF2", "NATF3"],
    "Natural Male": ["NATM0", "NATM1", "NATM2", "NATM3"],
    "Variety Female": ["VARF0", "VARF1", "VARF2", "VARF3", "VARF4"],
    "Variety Male": ["VARM0", "VARM1", "VARM2", "VARM3", "VARM4"],
}

ALL_VOICE_NAMES = [v for voices in ALL_VOICES.values() for v in voices]


def list_voices():
    """Print all available voices organized by category."""
    print("Available PersonaPlex voices:\n")
    for category, voices in ALL_VOICES.items():
        print(f"  {category}:")
        print(f"    {', '.join(voices)}")
    print(f"\n  Total: {len(ALL_VOICE_NAMES)} voices")
    print(f"\nUsage: python examples/custom_voice.py --input audio.wav --voices NATF2 NATM1")


def main():
    parser = argparse.ArgumentParser(
        description="Compare PersonaPlex voices on the same input audio."
    )
    parser.add_argument(
        "--list-voices",
        action="store_true",
        help="List all available voices and exit",
    )
    parser.add_argument(
        "--input",
        type=str,
        default="assets/test/input_assistant.wav",
        help="Path to input WAV file (default: assets/test/input_assistant.wav)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="voice_comparison/",
        help="Directory for output files (default: voice_comparison/)",
    )
    parser.add_argument(
        "--voices",
        nargs="+",
        type=str,
        help="Subset of voices to compare (default: all 18)",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default="You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way.",
        help="Text prompt for all voices",
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

    if args.list_voices:
        list_voices()
        sys.exit(0)

    # Determine which voices to use
    if args.voices:
        voices = []
        for v in args.voices:
            v_upper = v.upper()
            if v_upper not in ALL_VOICE_NAMES:
                print(f"Error: Unknown voice '{v}'. Use --list-voices to see options.", file=sys.stderr)
                sys.exit(1)
            voices.append(v_upper)
    else:
        voices = ALL_VOICE_NAMES

    # Validate
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

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:  {args.input}")
    print(f"Output: {output_dir}/")
    print(f"Voices: {', '.join(voices)} ({len(voices)} total)")
    print()

    # Heavy imports after validation
    import numpy as np
    import torch
    import sentencepiece
    import sphn
    from huggingface_hub import hf_hub_download
    from moshi.offline import (
        _get_voice_prompt_dir,
        seed_all,
        warmup,
        decode_tokens_to_pcm,
        wrap_with_system_tags,
        log,
    )
    from moshi.models import loaders, LMGen, MimiModel
    from moshi.models.lm import (
        load_audio as lm_load_audio,
        _iterate_audio as lm_iterate_audio,
        encode_from_sphn as lm_encode_from_sphn,
    )

    torch.set_grad_enabled(False)

    if args.seed is not None and args.seed != -1:
        seed_all(args.seed)

    device = args.device

    # --- Load models once ---
    print("Loading models (this takes a minute)...")
    t0 = time.time()

    hf_repo = loaders.DEFAULT_REPO
    hf_hub_download(hf_repo, "config.json")

    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=args.cpu_offload)
    lm.eval()

    frame_size = int(mimi.sample_rate / mimi.frame_rate)
    lm_gen = LMGen(
        lm,
        audio_silence_frame_cnt=int(0.5 * mimi.frame_rate),
        sample_rate=mimi.sample_rate,
        device=device,
        frame_rate=mimi.frame_rate,
        save_voice_prompt_embeddings=False,
        use_sampling=True,
        temp=0.8,
        temp_text=0.7,
        top_k=250,
        top_k_text=25,
    )

    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    voice_prompt_dir = _get_voice_prompt_dir(None, hf_repo)
    text_prompt_tokens = text_tokenizer.encode(wrap_with_system_tags(args.text_prompt))
    sample_rate = mimi.sample_rate

    load_time = time.time() - t0
    print(f"Models loaded in {load_time:.1f}s\n")

    # --- Process each voice ---
    for i, voice_name in enumerate(voices, 1):
        voice_prompt_path = os.path.join(voice_prompt_dir, f"{voice_name}.pt")
        if not os.path.exists(voice_prompt_path):
            print(f"  [{i}/{len(voices)}] Skipping {voice_name} — prompt file not found")
            continue

        print(f"  [{i}/{len(voices)}] Generating with {voice_name}...")
        t_start = time.time()

        # Reset streaming state
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        # Load voice prompt for this voice
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        lm_gen.text_prompt_tokens = text_prompt_tokens

        # Run system prompts
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        # Load user audio
        user_audio = lm_load_audio(args.input, sample_rate)
        total_target_samples = user_audio.shape[-1]

        # Generate
        generated_frames: List[np.ndarray] = []
        generated_text_tokens: List[str] = []

        for user_encoded in lm_encode_from_sphn(
            mimi,
            lm_iterate_audio(user_audio, sample_interval_size=lm_gen._frame_size, pad=True),
            max_batch=1,
        ):
            steps = user_encoded.shape[-1]
            for c in range(steps):
                step_in = user_encoded[:, :, c : c + 1]
                tokens = lm_gen.step(step_in)
                if tokens is None:
                    continue
                pcm = decode_tokens_to_pcm(mimi, other_mimi, lm_gen, tokens)
                generated_frames.append(pcm)
                text_token = tokens[0, 0, 0].item()
                if text_token not in (0, 3):
                    _text = text_tokenizer.id_to_piece(text_token).replace("\u2581", " ")
                    generated_text_tokens.append(_text)
                else:
                    text_token_map = ["EPAD", "BOS", "EOS", "PAD"]
                    generated_text_tokens.append(text_token_map[text_token])

        if not generated_frames:
            print(f"    Warning: No frames generated for {voice_name}, skipping.")
            continue

        # Concatenate and trim/pad
        output_pcm = np.concatenate(generated_frames, axis=-1)
        if output_pcm.shape[-1] > total_target_samples:
            output_pcm = output_pcm[:total_target_samples]
        elif output_pcm.shape[-1] < total_target_samples:
            pad_len = total_target_samples - output_pcm.shape[-1]
            output_pcm = np.concatenate(
                [output_pcm, np.zeros(pad_len, dtype=output_pcm.dtype)], axis=-1
            )

        # Write output
        out_wav = output_dir / f"{voice_name}.wav"
        out_json = output_dir / f"{voice_name}.json"
        sphn.write_wav(str(out_wav), output_pcm, sample_rate)
        with open(out_json, "w") as f:
            json.dump(generated_text_tokens, f, ensure_ascii=False)

        elapsed = time.time() - t_start
        print(f"    -> {out_wav.name} ({elapsed:.1f}s)")

    print(f"\nVoice comparison complete! Output in {output_dir}/")


if __name__ == "__main__":
    main()
