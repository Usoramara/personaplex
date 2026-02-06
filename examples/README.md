# PersonaPlex Examples

Ready-to-run example scripts for common PersonaPlex use cases. Each script validates prerequisites before loading the model.

## Prerequisites

1. **Install PersonaPlex:**
   ```bash
   pip install moshi/.
   ```

2. **Set your Hugging Face token** (after [accepting the license](https://huggingface.co/nvidia/personaplex-7b-v1)):
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

3. **GPU recommended** — all examples support `--device cpu` and `--cpu-offload` but GPU is strongly recommended for reasonable performance.

## Examples

### voice_assistant.py — Simple Voice Assistant

The simplest example. Takes a WAV file and produces a response using the default assistant persona.

```bash
# Use defaults (bundled test audio, NATF2 voice)
python examples/voice_assistant.py

# Specify input and voice
python examples/voice_assistant.py --input my_question.wav --output answer.wav --voice NATM1

# CPU with offloading
python examples/voice_assistant.py --input my_question.wav --device cpu --cpu-offload
```

| Flag | Default | Description |
|---|---|---|
| `--input` | `assets/test/input_assistant.wav` | Input WAV file |
| `--output` | `output_assistant.wav` | Output WAV file |
| `--output-text` | `output_assistant.json` | Output text JSON |
| `--voice` | `NATF2` | Voice name |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--cpu-offload` | off | Offload layers to CPU |
| `--seed` | `42424242` | Random seed (-1 to disable) |

---

### customer_service_bot.py — Customer Service Personas

Pre-built customer service personas with custom prompts. Includes 4 templates from the PersonaPlex prompting guide.

```bash
# List available personas
python examples/customer_service_bot.py --list-personas

# Use a built-in persona
python examples/customer_service_bot.py --persona restaurant --input call.wav

# Use a custom prompt
python examples/customer_service_bot.py \
  --custom-prompt "You work for TechFix and your name is Sarah..." \
  --input call.wav --voice NATF0
```

**Built-in personas:**

| ID | Business | Agent Name | Default Voice |
|---|---|---|---|
| `appliance_repair` | SwiftPlex Appliances | Farhod Toshmatov | NATM1 |
| `restaurant` | Jerusalem Shakshuka | Owen Foster | NATM0 |
| `drone_rental` | AeroRentals Pro | Tomaz Novak | NATM2 |
| `waste_management` | CitySan Services | Ayelen Lucero | NATF1 |

---

### batch_inference.py — Batch Processing

Process an entire directory of WAV files with a single model load. Much faster than processing files individually since model loading (~60s) happens only once.

```bash
# Process all WAVs in a directory
python examples/batch_inference.py --input-dir ./calls/ --output-dir ./responses/

# With custom voice and prompt
python examples/batch_inference.py \
  --input-dir ./calls/ \
  --output-dir ./responses/ \
  --voice NATM1 \
  --text-prompt "You work for Acme Corp..."
```

| Flag | Default | Description |
|---|---|---|
| `--input-dir` | required | Directory with input WAV files |
| `--output-dir` | required | Directory for outputs |
| `--voice` | `NATF2` | Voice name |
| `--text-prompt` | default assistant | Text prompt for all files |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--cpu-offload` | off | Offload layers to CPU |
| `--seed` | `42424242` | Random seed |

Output: `{output-dir}/{filename}.wav` and `{output-dir}/{filename}.json` for each input.

---

### custom_voice.py — Voice Comparison

Generate the same response with multiple voices for comparison. Loads the model once and loops over all selected voices.

```bash
# List all 18 voices
python examples/custom_voice.py --list-voices

# Compare all voices (default)
python examples/custom_voice.py --input assets/test/input_assistant.wav

# Compare specific voices
python examples/custom_voice.py --input my_audio.wav --voices NATF2 NATM1 VARF0 VARM2

# Custom output directory
python examples/custom_voice.py --input my_audio.wav --output-dir voice_samples/
```

| Flag | Default | Description |
|---|---|---|
| `--list-voices` | — | List voices and exit |
| `--input` | `assets/test/input_assistant.wav` | Input WAV file |
| `--output-dir` | `voice_comparison/` | Output directory |
| `--voices` | all 18 | Subset of voices to compare |
| `--text-prompt` | default assistant | Text prompt |
| `--device` | `cuda` | `cuda` or `cpu` |
| `--cpu-offload` | off | Offload layers to CPU |
| `--seed` | `42424242` | Random seed |

Output: `{output-dir}/{VOICE_NAME}.wav` and `{output-dir}/{VOICE_NAME}.json` for each voice.

## Available Voices

```
Natural Female:  NATF0, NATF1, NATF2, NATF3
Natural Male:    NATM0, NATM1, NATM2, NATM3
Variety Female:  VARF0, VARF1, VARF2, VARF3, VARF4
Variety Male:    VARM0, VARM1, VARM2, VARM3, VARM4
```
