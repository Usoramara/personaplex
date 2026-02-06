# Contributing

This repository is a fork of [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex). Core model changes and bug fixes should go upstream to NVIDIA's repository. This fork focuses on enhanced documentation, deployment guides, and usage examples.

## What We Welcome

- Documentation improvements and clarifications
- New example scripts demonstrating PersonaPlex use cases
- Deployment guides for additional platforms
- Bug reports and fixes for this fork's additions

## Reporting Issues

- **Model/core bugs:** Report to [NVIDIA/personaplex](https://github.com/NVIDIA/personaplex/issues)
- **Docs/examples/deployment issues:** Open an issue on this repository

## Development Setup

```bash
# Clone the repository
git clone https://github.com/justwybe/Nvidia-natural-conversation-model.git
cd Nvidia-natural-conversation-model

# Install system dependency
sudo apt install libopus-dev  # or: brew install opus (macOS)

# Install in development mode
pip install -e moshi/.

# Set your Hugging Face token
export HF_TOKEN=hf_your_token_here
```

## Submitting Changes

1. Fork this repository
2. Create a feature branch: `git checkout -b my-feature`
3. Make your changes
4. Test that example scripts parse correctly: `python examples/voice_assistant.py --help`
5. Submit a pull request with a clear description of the change

## Code Style

- Python: Follow the existing code style in `moshi/moshi/`
- Markdown: Use ATX-style headers (`#`), fenced code blocks with language tags
- Example scripts: Include argparse help, validate `HF_TOKEN` before heavy imports, support `--device` and `--cpu-offload` flags

## License

By contributing, you agree that your contributions will be licensed under the MIT License (see [LICENSE-MIT](LICENSE-MIT)).
