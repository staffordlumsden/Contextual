# Contextual

**Version:** v. 4.9 (06 July 2026)
**Author:** Stafford Lumsden  

Welcome to **Contextual**, a feature rich CLI for interacting with local and Hugging Face Large Language Models via Ollama v. 30 and above.

> **New in v. 4.9:** When selecting a chat model, choose **h** to paste an Ollama Hugging Face command such as `ollama run hf.co/{username}/{repository}:{quantization}`. Contextual extracts the `hf.co/...` model reference, normalizes pasted line breaks and quantization tag casing, pulls the model with Ollama, and then uses it for chat.

> **Speed note:** Contextual now warms the selected chat model on startup/model selection and keeps it alive for faster first-token response times. This moves the cold model-load wait before the first user prompt. Disable this behaviour with `CONTEXTUAL_PRELOAD_CHAT_MODEL=0 ./run_chatbot.sh`.

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
- 🔥 **Dynamic Model Switching**: Easily swap between chat models, embedding models, and Ollama-compatible Hugging Face chat models.
- 🧮 **Token Stats and Analytics**: View session stats with `/stats`, and toggle per-response Ollama metrics with `/set verbose`.
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
| `/set verbose`       | Toggle per-response Ollama analytics                    |
| `/set verbose on/off`| Explicitly enable or disable Ollama analytics            |

### Hugging Face Models

When Contextual asks you to select a chat model, local Ollama models are listed first. Select **h** to enter a Hugging Face model.

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
4. Contextual validates the reference, normalizes quantization tags to uppercase, and runs `ollama pull` so the model is available through the Ollama API before chat starts.

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
| `/set parameter <name> <value>`   | Set a specific Ollama runtime parameter                  |
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

### Presets
| Command              | Description                                               |
|----------------------|-----------------------------------------------------------|
| `/set cold`          | Apply the 'accuracy-cold' parameter preset               |
| `/set balanced`      | Apply the 'balanced' parameter preset                    |
| `/set warm`          | Apply the 'creative-warm' parameter preset               |

### Speed and Analytics

Contextual warms the selected chat model before the first prompt and after `/switch`, using Ollama keep-alive to reduce cold-start TTFT on the first real response. To skip model preloading:

```bash
CONTEXTUAL_PRELOAD_CHAT_MODEL=0 ./run_chatbot.sh
```

On Windows PowerShell:

```powershell
$env:CONTEXTUAL_PRELOAD_CHAT_MODEL="0"
.\run_chatbot.ps1
```

Use `/set verbose` to toggle an Ollama metrics box under each model response. The metrics include:

- First chunk time
- TTFT
- Input and output token counts
- Prompt eval time and prompt eval rate
- Generation time and output rate
- Ollama total time

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
