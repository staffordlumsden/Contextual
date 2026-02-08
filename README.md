# Contextual

**Version:** 3.3.3 (24 January 2026)  
**Author:** Stafford Lumsden  

Welcome to **Contextual**, a powerful, feature rich command line interface (CLI) designed to interact with locally deployed large language models (LLMs) run with Ollama 14.3 and above. Image generation implemented Jan '26.

---

## Install

1. Download/clone the repo
2. On macOS make `setup.sh` executable
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

Do something similar on Windows?

---

## Features

- 🎛️ **Interactive CLI**: Rich, TUI-style interface with multi-line prompt input, slash commands, and hotkeys.
- 📄 **Document Chat**: Load `.txt`, `.md`, `.docx`, `.pdf`, `.csv`, and `.xlsx` files, chunk them, and query with a chosen embedding model.
- 🔍 **RAG (Retrieval-Augmented Generation)**: Automatic document chunking, embedding, storage in ChromaDB, and context-aware answers.
- 🧠 **EmbeddingGemma**: Optimized support for `task: query embedding | query:` and `task: document embedding | text:` prefixes to improve semantic retrieval.
- 🔥 **Dynamic Model Switching**: Easily swap between chat models and embedding models.
- 🧮 **Token Stats**: View token usage and latency stats for each response.
- 📝 **Conversation Saving**: Export chats in Markdown or plain text.
- 🧩 **CSV Cleaning Mode**: Automatically clean and reformat CSVs for model-friendly analysis.
- 🖼️ **Image Generation**: Opens a new Terminal window to run `ollama run x/flux2-klein:9b` with your prompt and saves images to the selected folder. After generation, return to Contextual and press Enter to import and list the new PNGs.

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

2. **Pull Models:**
   ```bash
   ollama pull gemma3n
   ollama pull embeddinggemma
   ollama pull x/flux2-klein
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
3. Enter a prompt; Contextual opens a new Terminal window and runs:
   ```bash
   ollama run x/flux2-klein:9b --verbose --width <n> --height <n> --steps <n> --seed <n> "your prompt"
   ```
4. When the Terminal finishes and saves the image, return to Contextual and press Enter.
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

---

## License

MIT License.  
Copyright © 2026 Stafford Lumsden.
