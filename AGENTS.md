# AGENTS.md

Context
- Project: Contextual (contextual2.py) CLI for Ollama-based chat, RAG, and CSV cleaning.
- Date: 20 November 2026.
- Scope: Chat through Ollama, LiteRT-LM and Apple's on-device Foundation Models, document RAG and CSV preparation.

Current requirements
- Image generation was removed at the user's request. Do not restore it without a new request.
- Startup modes: General chat, Chat with a document, Clean CSV, Read the README.
- Interface: v. 7.0 [Interface] (20 November 2026), with Seoul256 colours, light/dark themes and borderless shaded sections.
- Preserve existing local launcher changes for symbolic-link resolution.
- Downloaded Ollama model files are managed separately from this application.

Validation
- Run `venv/bin/python -m unittest discover -s tests` and `git diff --check` after relevant changes.
