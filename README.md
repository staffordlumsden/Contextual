# Contextual

**Version:** v. 5.0 (19 October 2026)
**Author:** Stafford Lumsden  
**Project site:** https://staffordlumsden.github.io/Contextual/  

Welcome to **Contextual**, a feature rich CLI for interacting with local models served by Ollama and LiteRT-LM. Ollama-compatible Hugging Face models can also be imported from the chat-model picker.

> **New in v. 5.0:** The chat-model picker now discovers and groups models served by Ollama and LiteRT-LM. Selecting a model selects its backend automatically. System prompts, thinking controls, presets, parameter commands, streaming and response statistics use the selected backend. Parameters unsupported by that backend are identified rather than silently passed through.

> **New in v. 4.9:** When selecting a chat model, choose **h** to paste an Ollama Hugging Face command such as `ollama run hf.co/{username}/{repository}:{quantization}`. Contextual extracts the `hf.co/...` model reference, normalizes pasted line breaks and quantization tag casing, pulls the model with live Ollama progress, and then uses it for chat.

> **Speed note:** Contextual warms the selected chat model on startup and after `/switch`, moving the cold model-load wait before the first user prompt. Disable this behaviour with `CONTEXTUAL_PRELOAD_CHAT_MODEL=0 ./run_chatbot.sh`.

> **Windows support note:** Contextual now includes PowerShell setup and runner scripts (`setup_windows.ps1` and `run_chatbot.ps1`). Chat, RAG, CSV, and image generation are intended to run on Windows with Ollama for Windows; image generation opens PowerShell on Windows, Terminal on macOS, and a detected terminal emulator on Linux.

---

## Install

### macOS and Linux

1. Download/clone the repo
2. Make `setup.sh` executable
```bash
chmod +x /path/to/setup.sh
``` 
3. Run 
```bash
./setup.sh
```
This will download and install dependencies 

4. Either make `run_chatbot.sh` executable and run it, or use 
 ```bash
   python contextual2.py
 ```
5. Add to PATH as needed.

### Windows

1. Download/clone the repo and open PowerShell in the project folder.
2. If PowerShell blocks local scripts, allow them for the current session:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```
3. Run setup:
```powershell
.\setup_windows.ps1
```
4. Run Contextual:
```powershell
.\run_chatbot.ps1
```

The Windows setup script creates `venv`, installs dependencies, and skips Unix-only `uvloop`.

### Banner Font

Contextual uses `pyfiglet` for the startup banner. The default font is `4max`.

Use any bundled pyfiglet font with:
```bash
./run_chatbot.sh --figlet-font slant
```

Or set it with an environment variable:
```bash
CONTEXTUAL_FIGLET_FONT=slant ./run_chatbot.sh
```

List available fonts with:
```bash
./run_chatbot.sh --list-figlet-fonts
```

---

## Features

- 🎛️ **Interactive CLI**: Rich, TUI-style interface with multi-line prompt input, slash commands, and hotkeys.
- 📄 **Document Chat**: Load `.txt`, `.md`, `.docx`, `.pdf`, `.csv`, and `.xlsx` files, chunk them, and query with a chosen embedding model.
- 🔍 **RAG (Retrieval-Augmented Generation)**: [Chonkie](https://github.com/feyninc/chonkie)-backed recursive document chunking, embedding, storage in ChromaDB, and context-aware answers.
- 🧠 **EmbeddingGemma**: Optimized support for `task: query embedding | query:` and `task: document embedding | text:` prefixes to improve semantic retrieval.
- 🔥 **Backend-aware Model Switching**: Select from grouped Ollama and LiteRT-LM models, or import an Ollama-compatible Hugging Face model.
- 🧮 **Token Stats and Analytics**: View session stats with `/stats`, and toggle backend response metrics with `/set verbose`.
- 📝 **Conversation Saving**: Export chats in Markdown or plain text.
- 🧩 **CSV Cleaning Mode**: Automatically clean and reformat CSVs for model-friendly analysis.
- 🖼️ **Image Generation**: Opens a platform terminal window to run `ollama run x/flux2-klein:9b` with your prompt and saves images to the selected folder. After generation, return to Contextual and press Enter to import and list the new PNGs.

---

## EmbeddingGemma Performance Overview

- **Compact, efficient architecture**: ~300M parameters, optimized for on-device environments with quantization.  
- **Minimal RAM footprint**: Operates within <200MB RAM when quantized.  
- **Low latency on EdgeTPU**: Embedding generation in <15ms for ~256 tokens.  
- **Top-tier benchmark performance**: Highest-scoring multilingual embedding model under 500M parameters on MTEB.  
- **Multilingual and customizable**: Supports 100+ languages; 768-dim output by default, adjustable to 512/256/128 dims; handles up to 2048 tokens.

---

## Full Install

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   On Windows, prefer `.\setup_windows.ps1` because it skips Unix-only `uvloop`.

2. **Pull Models:**
   ```bash
   ollama pull gemma3n
   ollama pull embeddinggemma
   ollama pull x/flux2-klein
   ```

   You can also run Ollama-compatible Hugging Face models directly from the chat model picker by selecting **h** and pasting:
   ```bash
   ollama run hf.co/{username}/{repository}:{quantization}
   ```

   LiteRT-LM models appear automatically when its OpenAI-compatible server is running. Contextual checks `http://127.0.0.1:9379/v1` by default:
   ```bash
   litert-lm serve --host 127.0.0.1 --port 9379
   ```

   To use another LiteRT-LM endpoint:
   ```bash
   CONTEXTUAL_LITERT_BASE_URL=http://127.0.0.1:8080/v1 ./run_chatbot.sh
   ```

   Contextual budgets LiteRT-LM conversation history against an 8,192-token context window by default. If a served model uses a different limit, set `CONTEXTUAL_LITERT_CONTEXT_TOKENS` to match it.

3. **Run Contextual:**
   ```bash
   python contextual2.py
   ```

---

## Commands

### General
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/help`              | Display this help message                                |
| `/stats`             | Display session statistics                               |
| `/save`              | Save the current conversation to a file                  |
| `/new`               | Start a new chat session                                 |

### Model
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/switch`            | Switch to a different chat model                         |
| `/prompt`            | Set a custom system prompt for the session               |
| `/set think`         | Enable thinking mode for supported models                |
| `/set nothink`       | Disable thinking mode                                    |
| `/set verbose`       | Toggle per-response backend analytics                   |
| `/set verbose on/off`| Explicitly enable or disable backend analytics           |

### Chat Model Backends

The model picker is divided into:

- **Ollama served models** discovered from the local Ollama service.
- **LiteRT-LM served models** discovered from its OpenAI-compatible `/v1/models` endpoint.
- **Additional model**, where **h** imports an Ollama-compatible Hugging Face model into Ollama.

The selected model carries its backend through general chat, document chat and `/switch`. Ollama continues to supply document embeddings even when LiteRT-LM generates the answer.

### Hugging Face Models

When Contextual asks you to select a chat model, Ollama and LiteRT-LM models are listed in separate groups. Select **h** to enter a Hugging Face model that Ollama will pull and serve.

1. Open an Ollama-compatible model page on Hugging Face.
2. Copy the Ollama command, for example:
   ```bash
   ollama run hf.co/{username}/{repository}:{quantization}
   ```
3. Paste the full command into Contextual, then press Enter on a blank line to submit. You may also paste only:
   ```bash
   hf.co/{username}/{repository}:{quantization}
   ```
   Contextual also accepts `owner/repository:{quantization}` and Hugging Face URLs.
4. Contextual validates the reference, normalizes quantization tags to uppercase, and runs `ollama pull` with live progress so the model is available through the Ollama API before chat starts.

Example:
```bash
ollama run hf.co/Andycurrent/Gemma-3-1B-it-GLM-4.7-Flash-Heretic-Uncensored-Thinking_GGUF:Q6_K
```

This option is also available from `/switch` during a chat session.

### File
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/file`              | Switch to a different document for chat                  |
| `/csv`               | Clean and load a CSV file for chat                       |
| `/chunks N`          | Set the number of document chunks to retrieve            |

### Decoding & Runtime
| Command                            | Description                                               |
|-----------------------------------|-----------------------------------------------------------|
| `/set parameter <name> <value>`   | Set a parameter supported by the selected backend        |
| `seed <int>`                      | Random number seed                                       |
| `num_predict <int>`               | Max number of tokens to predict                          |
| `top_k <int>`                     | Pick from top k num of tokens                            |
| `top_p <float>`                   | Pick token based on sum of probabilities                 |
| `min_p <float>`                   | Pick token based on top token probability * min_p        |
| `num_ctx <int>`                   | Set the context size                                     |
| `temperature <float>`             | Set creativity level                                     |
| `repeat_penalty <float>`          | How strongly to penalize repetitions                     |
| `repeat_last_n <int>`             | Set how far back to look for repetitions                 |
| `num_gpu <int>`                   | The number of layers to send to the GPU                  |
| `stop <string>`                   | Set the stop parameter                                   |

Contextual uses the same commands for both backends and translates them where necessary. LiteRT-LM supports `seed`, `num_predict`, `top_k`, `top_p` and `temperature`; `num_predict` is sent as `max_completion_tokens`. When `num_predict` has not been set, Contextual limits LiteRT-LM responses to 1,024 tokens to prevent a looping generation from occupying the local server. Set `CONTEXTUAL_LITERT_MAX_COMPLETION_TOKENS` to change that default, or use `/set parameter num_predict <value>` for the current session. Ollama-specific settings remain available when an Ollama model is selected. `/help` lists only the parameters supported by the current backend.

### Presets
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/set cold`          | Apply the 'accuracy-cold' parameter preset               |
| `/set balanced`      | Apply the 'balanced' parameter preset                    |
| `/set warm`          | Apply the 'creative-warm' parameter preset               |

### Speed and Analytics

Contextual warms the selected chat model before the first prompt and after `/switch`. Ollama uses keep-alive; LiteRT-LM receives a short warm-up request. To skip model preloading:

```bash
CONTEXTUAL_PRELOAD_CHAT_MODEL=0 ./run_chatbot.sh
```

On Windows PowerShell:

```powershell
$env:CONTEXTUAL_PRELOAD_CHAT_MODEL="0"
.\run_chatbot.ps1
```

Use `/set verbose` to toggle a backend metrics box under each model response. Available metrics include:

- First chunk time
- TTFT
- Input and output token counts
- Prompt eval time and prompt eval rate
- Generation time and output rate
- Backend-reported evaluation and total time, when available

### Image Generation
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/set width <n>`     | Set image width in pixels                                |
| `/set height <n>`    | Set image height in pixels                               |
| `/set steps <n>`     | Set diffusion steps (fewer is faster, more adds detail)  |
| `/set seed <n>`      | Set a random seed for reproducible results               |

#### How Image Generation Works
1. Select **Image generation** from the startup mode picker.
2. Choose a folder where images should be saved.
3. Enter a prompt; Contextual opens a platform terminal window and runs:
   ```bash
   ollama run x/flux2-klein:9b --verbose --width <n> --height <n> --steps <n> --seed <n> "your prompt"
   ```
4. When the terminal finishes and saves the image, return to Contextual and press Enter.
5. Contextual scans the selected folder and lists newly created PNG files.

### Input
| Key Combination      | Description                                               |
|----------------------|-----------------------------------------------------------|
| `[Esc] + [Enter]`    | Submit a multi-line prompt or a slash command            |
| `[Ctrl] + [C]`       | Interrupt the bot while it is thinking                   |

### Exit
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/exit` or `/quit`   | Exit the application                                     |

---

## Embedding Prompt Prefixes

When using **EmbeddingGemma**, prepend these simple prefixes to clearly define your embedding type:

- **Document Chunks**:
  ```
  task: document embedding | text: [chunk content]
  ```

- **Queries**:
  ```
  task: query embedding | query: [your search query]
  ```

These prefixes improve retrieval accuracy in RAG workflows.

Contextual applies these prefixes automatically for EmbeddingGemma during document ingestion and query retrieval.

## Document Chunking

Document chat uses [Chonkie](https://github.com/feyninc/chonkie)'s recursive chunker with overlap refinement before storing chunks in ChromaDB. This keeps chunks closer to natural paragraph and sentence boundaries while preserving cross-chunk context for retrieval. Chonkie runs automatically in the background when a document is loaded; users do not need to run a separate command.

---

## License

MIT License.  
Copyright © 2026 Stafford Lumsden.
