#!/usr/bin/env python3
"""Batch inference example — process multiple WAV files with a single model load.

Loads the PersonaPlex model once, then processes each WAV file in a directory.
This is significantly faster than calling run_inference() per file, since model
loading takes the majority of the time.

Usage:
    python examples/batch_inference.py --input-dir ./calls/ --output-dir ./responses/
    python examples/batch_inference.py --input-dir ./calls/ --output-dir ./responses/ --voice NATM1 --text-prompt "You work for ..."
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional

DEFAULT_TEXT_PROMPT = "You are a wise and friendly teacher. Answer questions or provide advice in a clear and engaging way."


def main():
    parser = argparse.ArgumentParser(
        description="Batch inference — process multiple WAVs with one model load."
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input WAV files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory for output WAV and JSON files",
    )
    parser.add_argument(
        "--voice",
        type=str,
        default="NATF2",
        help="Voice name (default: NATF2)",
    )
    parser.add_argument(
        "--text-prompt",
        type=str,
        default=DEFAULT_TEXT_PROMPT,
        help="Text prompt for all files",
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

    # Validate
    if not os.environ.get("HF_TOKEN"):
        print(
            "Error: HF_TOKEN environment variable not set.\n"
            "1. Accept the model license: https://huggingface.co/nvidia/personaplex-7b-v1\n"
            "2. Set your token: export HF_TOKEN=hf_...",
            file=sys.stderr,
        )
        sys.exit(1)

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        print(f"Error: Input directory not found: {input_dir}", file=sys.stderr)
        sys.exit(1)

    wav_files = sorted(input_dir.glob("*.wav"))
    if not wav_files:
        print(f"Error: No .wav files found in {input_dir}", file=sys.stderr)
        sys.exit(1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(wav_files)} WAV files in {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Voice: {args.voice}")
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

    # Mimi (two instances needed)
    mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
    mimi = loaders.get_mimi(mimi_weight, device)
    other_mimi = loaders.get_mimi(mimi_weight, device)

    # Tokenizer
    tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
    text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

    # Moshi LM
    moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
    lm = loaders.get_moshi_lm(moshi_weight, device=device, cpu_offload=args.cpu_offload)
    lm.eval()

    # LMGen
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

    # Streaming mode
    mimi.streaming_forever(1)
    other_mimi.streaming_forever(1)
    lm_gen.streaming_forever(1)

    # Warmup
    warmup(mimi, other_mimi, lm_gen, device, frame_size)

    # Voice prompt
    voice_prompt_dir = _get_voice_prompt_dir(None, hf_repo)
    voice_prompt_path = os.path.join(voice_prompt_dir, f"{args.voice}.pt")
    if not os.path.exists(voice_prompt_path):
        print(f"Error: Voice prompt not found: {voice_prompt_path}", file=sys.stderr)
        sys.exit(1)

    # Text prompt tokens
    text_prompt_tokens = text_tokenizer.encode(wrap_with_system_tags(args.text_prompt))

    load_time = time.time() - t0
    print(f"Models loaded in {load_time:.1f}s\n")

    # --- Process each file ---
    sample_rate = mimi.sample_rate
    total_time = 0.0

    for i, wav_path in enumerate(wav_files, 1):
        print(f"[{i}/{len(wav_files)}] Processing {wav_path.name}...")
        t_start = time.time()

        # Reset streaming state between conversations
        mimi.reset_streaming()
        other_mimi.reset_streaming()
        lm_gen.reset_streaming()

        # Load voice prompt and text prompt
        lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
        lm_gen.text_prompt_tokens = text_prompt_tokens

        # Run system prompts (voice -> silence -> text -> silence)
        lm_gen.step_system_prompts(mimi)
        mimi.reset_streaming()

        # Load user audio
        user_audio = lm_load_audio(str(wav_path), sample_rate)
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
            print(f"  Warning: No frames generated for {wav_path.name}, skipping.")
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

        # Write outputs
        stem = wav_path.stem
        out_wav = output_dir / f"{stem}.wav"
        out_json = output_dir / f"{stem}.json"
        sphn.write_wav(str(out_wav), output_pcm, sample_rate)
        with open(out_json, "w") as f:
            json.dump(generated_text_tokens, f, ensure_ascii=False)

        elapsed = time.time() - t_start
        total_time += elapsed
        audio_duration = total_target_samples / sample_rate
        print(f"  -> {out_wav.name} ({elapsed:.1f}s, {audio_duration:.1f}s audio)")

    print(f"\nBatch complete: {len(wav_files)} files in {total_time:.1f}s (+ {load_time:.1f}s model loading)")


if __name__ == "__main__":
    main()
