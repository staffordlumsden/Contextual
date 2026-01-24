# AGENTS.md

Context
- Project: Contextual (contextual2.py) CLI for Ollama-based chat, RAG, and CSV cleaning.
- Date: 24 January 2026.
- Goal: Add image generation support using x/flux2-klein with a new startup mode.

Work completed
- Added Image generation mode with folder picker, prompt loop, progress spinner, and image save flow.
- Implemented /set width, /set height, /set steps, and /set seed for image parameters.
- Reordered startup mode picker (General chat, Image generation, Chat with a document, Clean CSV, Read README).
- Updated banner version/welcome text and README for new capabilities.
- Defaulted image model to x/flux2-klein:9b and added model resolution fallback for installed flux variants.
- Added image generation fallback to stream mode to avoid JSON “Extra data” parsing errors.
- Set default steps to 4 for flux2-klein models.
- Added macOS Terminal launch fallback for image generation to run `ollama run` externally in image mode only.
