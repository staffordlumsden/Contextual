#!/usr/bin/env python3  
import ollama  
import pandas as pd  
import os  
import docx  
from PyPDF2 import PdfReader  
from rich.console import Console  
from rich.panel import Panel  
from rich.markdown import Markdown  
from rich.prompt import Prompt, IntPrompt  
from rich.align import Align  
from rich.columns import Columns  
from rich.table import Table  
from rich.text import Text  
from rich.rule import Rule  
from rich.spinner import Spinner  
import sys  
import argparse  
import pyfiglet  
import time  
from datetime import datetime  
import io  
from pick import pick  
import chromadb  
import re  
import base64  
import json  
import urllib.request  
import urllib.error  
import subprocess  
import tempfile  
import shutil  
import shlex  
from rich.progress import track, ProgressBar  
from rich.live import Live  
from rich.console import Group  
from prompt_toolkit import PromptSession  
from prompt_toolkit.key_binding import KeyBindings  
from prompt_toolkit.filters import Condition  
from prompt_toolkit.application import get_app  
import threading  
import queue

# --- Ollama embeddings compatibility helper ---
def embed_text(model: str, text: str):
    """
    Calls ollama.embeddings with the right argument name across client versions.
    Newer clients accept 'input=', older ones require 'prompt='.
    """
    try:
        # Newer clients
        return ollama.embeddings(model=model, input=text)
    except TypeError:
        # Older clients
        return ollama.embeddings(model=model, prompt=text)
    
VALID_OLLAMA_PARAMETERS = {  
    "seed": int,  
    "num_predict": int,  
    "top_k": int,  
    "top_p": float,  
    "min_p": float,  
    "num_ctx": int,  
    "temperature": float,  
    "repeat_penalty": float,  
    "repeat_last_n": int,  
    "num_gpu": int,  
    "stop": list,  
}  
  
OLLAMA_PARAMETER_PRESETS = {  
    "cold": {  
        "temperature": 0.35,  
        "top_p": 0.85,  
        "num_predict": 1024,  
        "repeat_penalty": 1.10,  
    },  
    "balanced": {  
        "temperature": 0.70,  
        "top_p": 0.97,  
        "num_predict": 768,  
        "repeat_penalty": 1.15,  
    },  
    "warm": {  
        "temperature": 0.95,  
        "top_p": 0.95,  
        "num_predict": 768,  
        "repeat_penalty": 1.05,  
    },  
}  
  
# Create a reusable session  
session = PromptSession()  
  
# Setup key bindings for multiline input  
bindings = KeyBindings()  
  
@bindings.add('c-d')  
def _(event):  
    " Exit when `c-d` is pressed. "  
    event.app.exit()  
  
# Custom condition to check if the current line is empty.  
submit_on_empty_line = Condition(lambda: get_app().current_buffer.document.current_line == '')  
  
@bindings.add('enter', filter=submit_on_empty_line)  
def _(event):  
    """  
    Submit input when Enter is pressed on an empty line.  
    """  
    event.current_buffer.validate_and_handle()  
  
@bindings.add('escape', 'enter')  
def _(event):  
    """  
    Submit input on Option+Enter.  
    """  
    event.current_buffer.validate_and_handle()  
  
# Ensure stdout and stderr are configured for UTF-8  
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')  
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')  
  
console = Console()  
IMAGE_MODEL = "x/flux2-klein:9b"  
IMAGE_DEFAULT_WIDTH = 1024  
IMAGE_DEFAULT_HEIGHT = 1024  
  
  
def print_banner():  
    """Prints a colorful, responsive Figlet banner for the chatbot."""  
    console.print("\n\n")  
    width = console.width  
    if width < 80:  # Threshold for when to switch to a simpler banner  
        console.print(Panel(Align.center("Contextual"), style="bold white", border_style="white"))  
    else:  
        banner_text = pyfiglet.figlet_format("Contextual", font="4Max", width=int(width * 0.9))  
        lines = banner_text.split('\n')  
        rainbow_colors = [  
            "bold bright_red",  
            "bold bright_yellow",  
            "bold bright_green",  
            "bold bright_cyan",  
            "bold bright_blue",  
            "bold bright_magenta"  
        ]  
        for i, line in enumerate(lines):  
            color = rainbow_colors[i % len(rainbow_colors)]  
            console.print(Align.center(f"[{color}]{line}[/{color}]"))  
      
    console.print()  
    console.print(Align.center(Text("©2026 Stafford Lumsden v. 3.3.3 (24 January 2026)", style="white")), highlight=False)  
    console.print("\n\n")  
    console.print(Panel(Align.center("Welcome to Contextual, a powerful, feature rich command line interface (CLI) designed to interact with locally deployed large language models (LLMs) run with Ollama 14.3 and above. Image generation implemented Jan '26"), style="bold white", border_style="white"))  
  
def print_help():  
    """Prints a responsive, color-coded help message with available commands."""  
    table = Table.grid(expand=True, padding=(0, 2))  
    table.add_column(width=20)  # Command column  
    table.add_column()          # Description column  
  
    # Helper to add a section  
    def add_section(title, color, commands):  
        table.add_row(Text(title, style=f"bold {color}"), "")  
        for command, description in commands:  
            table.add_row(Text(command, style="white"), Text(description, style="white"))  
        table.add_row() # Spacer  
  
    # General Section  
    add_section("General", "green", [  
        ("/help", "Display this help message"),  
        ("/stats", "Display session statistics"),  
        ("/save", "Save the current conversation to a file"),  
        ("/new", "Start a new chat session"),  
    ])  
  
    # Model Section  
    add_section("Model", "grey50", [  
        ("/switch", "Switch to a different chat model"),  
        ("/prompt", "Set a custom system prompt for the session"),  
        ("/set think", "Enable thinking mode for supported models"),  
        ("/set nothink", "Disable thinking mode"),  
    ])  
  
    # File Section  
    add_section("File", "cyan", [  
        ("/file", "Switch to a different document for chat"),  
        ("/csv", "Clean and load a CSV file for chat"),  
        ("/chunks N", "Set the number of document chunks to retrieve"),  
    ])  
  
    # Image Section  
    add_section("Image", "bright_blue", [  
        ("/set width <n>", "Set image width in pixels (image mode)"),  
        ("/set height <n>", "Set image height in pixels (image mode)"),  
        ("/set steps <n>", "Set image diffusion steps (image mode)"),  
        ("/set seed <n>", "Set image random seed (image mode)"),  
    ])  
  
    # Decoding & Runtime Section  
    table.add_row(Text("Decoding & Runtime", style="bold yellow"), "")  
    table.add_row(Text("/set parameter <name> <value>", style="white"), Text("Set a specific Ollama runtime parameter", style="white"))  
    table.add_row(Text("  seed <int>", style="white"), Text("Random number seed", style="white"))  
    table.add_row(Text("  num_predict <int>", style="white"), Text("Max number of tokens to predict", style="white"))  
    table.add_row(Text("  top_k <int>", style="white"), Text("Pick from top k num of tokens", style="white"))  
    table.add_row(Text("  top_p <float>", style="white"), Text("Pick token based on sum of probabilities", style="white"))  
    table.add_row(Text("  min_p <float>", style="white"), Text("Pick token based on top token probability * min_p", style="white"))  
    table.add_row(Text("  num_ctx <int>", style="white"), Text("Set the context size", style="white"))  
    table.add_row(Text("  temperature <float>", style="white"), Text("Set creativity level", style="white"))  
    table.add_row(Text("  repeat_penalty <float>", style="white"), Text("How strongly to penalize repetitions", style="white"))  
    table.add_row(Text("  repeat_last_n <int>", style="white"), Text("Set how far back to look for repetitions", style="white"))  
    table.add_row(Text("  num_gpu <int>", style="white"), Text("The number of layers to send to the GPU", style="white"))  
    table.add_row(Text("  stop <string>", style="white"), Text("Set the stop parameter", style="white"))  
    table.add_row() # Spacer  
  
    # Presets Section  
    add_section("Presets", "light_purple", [  
        ("/set cold", "Apply the 'accuracy-cold' parameter preset"),  
        ("/set balanced", "Apply the 'balanced' parameter preset"),  
        ("/set warm", "Apply the 'creative-warm' parameter preset"),  
    ])  
  
    # Input Section  
    add_section("Input", "magenta", [  
        ("Use [Esc] + [Enter]", "Submit a multi-line prompt or a slash command"),  
        ("Use [Ctrl] + [C]", "Interrupt the bot while it is thinking"),  
    ])  
  
    # Exit Section  
    add_section("Exit", "red", [  
        ("/exit or /quit", "Exit the application"),  
    ])  
  
    console.print(Panel(table, title="[bold]Help Menu[/bold]", border_style="white", expand=True))  
  
def print_stats(start_time, model, file_name, turns, input_tokens, output_tokens, ollama_parameters, is_document_chat):  
    """Prints the session statistics and model parameters side-by-side."""  
    session_time = time.time() - start_time  
    mode = "Document Chat" if is_document_chat else "General Chat"  
    stats_text = f"""  
    Date: {datetime.now().strftime('%Y-%m-%d')}  
    Time: {datetime.now().strftime('%H:%M:%S')}  
    Session Time: {session_time:.2f} seconds  
    Mode: {mode}  
    Model: {model}  
    File: {file_name if file_name else 'N/A'}  
    Turns: {turns}  
    Input Tokens: {input_tokens}  
    Output Tokens: {output_tokens}  
    """  
      
    params_text = "\n".join([f"{key}: {value}" for key, value in ollama_parameters.items()])  
    if not params_text:  
        params_text = "No parameters set."  
  
    stats_panel = Panel(stats_text, title="[bold green]Session Statistics[/bold green]", border_style="white", style="white")  
    params_panel = Panel(params_text, title="[bold yellow]Model Parameters[/bold yellow]", border_style="yellow", style="yellow")  
  
    console.print(Columns([stats_panel, params_panel]))  
  

def read_file_content(file_path):  
    """Reads content from various file types."""  
    file_path = file_path.strip().strip('"\''"")  
    if not os.path.exists(file_path):  
        raise FileNotFoundError(f"No such file or directory: '{file_path}'")  
  
    ext = os.path.splitext(file_path)[1].lower()  
    if ext == '.csv':  
        return pd.read_csv(file_path).to_string()  
    elif ext == '.xlsx':  
        return pd.read_excel(file_path).to_string()  
    elif ext in ['.txt', '.md']:  
        with open(file_path, 'r', encoding='utf-8') as f:  
            return f.read()  
    elif ext == '.docx':  
        doc = docx.Document(file_path)  
        return "\n".join([para.text for para in doc.paragraphs])  
    elif ext == '.pdf':  
        # EDIT: make PDF extraction robust to None returns
        reader = PdfReader(file_path)  
        texts = []  
        for page in reader.pages:  
            t = page.extract_text() or ""  
            texts.append(t)  
        return "\n".join(texts)  
    else:  
        raise ValueError("Unsupported file type. Please use .csv, .xlsx, .txt, .md, .docx, or .pdf.")  

# EDIT: add a simple overlapping chunker instead of split('\n\n')
def chunk_text(txt, size=4000, overlap=400):
    out, i = [], 0
    n = len(txt)
    while i < n:
        out.append(txt[i:i+size])
        i += max(1, size - overlap)
    return [c for c in out if c.strip()]

def select_embedding_model(is_interactive):  
    """Lists available embedding models and prompts the user to select one."""  
    try:  
        models_info = ollama.list()  
        # EDIT: detect both "embed" and "embedding" substrings
        available_models = [model['model'] for model in models_info['models'] 
                            if any(s in model['model'].lower() for s in ('embed', 'embedding'))]  
        if not available_models:  
            console.print(Panel("[bold red]No embedding models found. Please install one (e.g., 'ollama pull nomic-embed-text').[/bold red]"))  
            return None  
  
        if is_interactive:  
            default_model = "nomic-embed-text"  
            if default_model in available_models:  
                available_models.insert(0, available_models.pop(available_models.index(default_model)))  
              
            model_list = "\n".join(f"[yellow]{i + 1}[/yellow]: {model_name}" for i, model_name in enumerate(available_models))  
            console.print(Panel(model_list, title="[bold green]Select an Embedding Model[/bold green]", expand=True))  
            choice = IntPrompt.ask("Select a model", choices=[str(i+1) for i in range(len(available_models))], default=1)  
            return available_models[choice - 1]  
        else:  
            return "nomic-embed-text"  
  
    except Exception as e:  
        console.print(Panel(f"[bold red]Error fetching embedding models: {e}[/bold red]"))  
        return None  
  
def select_chat_model(is_interactive):  
    """Lists available chat models and prompts the user to select one."""  
    try:  
        models_info = ollama.list()  
        available_models = [model['model'] for model in models_info['models'] if 'embed' not in model['model']]  
        if not available_models:  
            console.print(Panel("[bold red]No chat models found. Please install one (e.g., 'ollama pull llama2').[/bold red]"))  
            return None  
  
        if is_interactive:  
            model_list = "\n".join(f"[yellow]{i + 1}[/yellow]: {model_name}" for i, model_name in enumerate(available_models))  
            console.print(Panel(model_list, title="[bold green]Select a Chat Model[/bold green]", expand=True))  
            choice = IntPrompt.ask("Select a model", choices=[str(i+1) for i in range(len(available_models))])  
            return available_models[choice - 1]  
        else:  
            return available_models[0]  
  
    except Exception as e:  
        console.print(Panel(f"[bold red]Error fetching chat models: {e}[/bold red]"))  
        return None  
  
def sanitize_collection_name(name):  
    """Sanitizes a string to be a valid ChromaDB collection name."""  
    name = re.sub(r'[^a-zA-Z0-9._-]', '_', name)  
    if len(name) < 3:  
        name = name + '___'  
    if len(name) > 63:  
        name = name[:63]  
    if not re.match(r'^[a-zA-Z0-9]', name):  
        name = 'a' + name[1:]  
    if not re.match(r'.*[a-zA-Z0-9]$', name):  
        name = name[:-1] + 'a'  
    return name  
  
def create_chroma_collection(file_path, file_content, embedding_model):  
    """Creates a ChromaDB collection and stores the document embeddings."""  
    # EDIT: make persistent so we don't re-embed every run
    client = chromadb.PersistentClient(path=".chroma")  
    base_name = os.path.splitext(os.path.basename(file_path))[0]  
    collection_name = sanitize_collection_name(base_name)  
      
    try:  
        client.delete_collection(name=collection_name)  
    except Exception:  
        pass  
    collection = client.get_or_create_collection(collection_name)  
  
    # EDIT: use overlapping chunker
    chunks = chunk_text(file_content)  
      
    # Batch processing for embeddings  
    embeddings = []  
    for chunk in track(chunks, description="[bold green]Generating embeddings...[/bold green]"):  
        # EDIT: embeddings API uses input= not prompt=
        response = embed_text(model=embedding_model, text=chunk)  
        embeddings.append(response["embedding"])  
  
    # Batch add to ChromaDB  
    if embeddings:  
        # EDIT: add metadatas so we can cite source/chunk
        collection.add(  
            ids=[str(i) for i in range(len(chunks))],  
            embeddings=embeddings,  
            documents=chunks,  
            metadatas=[{"source": os.path.basename(file_path), "chunk": i} for i in range(len(chunks))]  
        )  
    return collection, len(chunks)  
  
def select_and_load_file(file_path=None, embedding_model=None):  
    """Handles file selection, loading, and embedding."""  
    if file_path is None and not sys.stdout.isatty():  
        console.print(Panel("[bold red]Error: No file path provided in non-interactive mode.[/bold red]"))  
        return None, None, None  
  
    while True:  
        if file_path is None:  
            if sys.stdout.isatty():  
                file_path = open_file_picker()  
                if not file_path:  
                    console.print(Panel("[bold yellow]No file selected. Returning to main menu.[/bold yellow]"))  
                    return None, None, None  
            else:  
                console.print(Panel("[bold red]Error: No file path provided in non-interactive mode.[/bold red]"))  
                return None, None, None  
          
        try:  
            content = read_file_content(file_path)  
            collection, num_chunks = create_chroma_collection(file_path, content, embedding_model)  
              
            file_basename = os.path.basename(file_path)  
            console.print(Panel(f"[bold green]Successfully loaded:[/] [white]{file_basename}[/white] and divided into [bold red]{num_chunks}[/bold red] chunks.", expand=False))  
            return content, file_path, collection  
        except (FileNotFoundError, ValueError, chromadb.errors.InvalidArgumentError) as e:  
            console.print(Panel(f"[bold red]Error: {e}[/bold red]"))  
            if not sys.stdout.isatty() or file_path:  
                return None, None, None  
            file_path = None  
  
def open_file_picker():  
    """Displays an interactive file picker to browse and select a document."""  
    supported_extensions = ['.csv', '.xlsx', '.txt', '.md', '.docx', '.pdf']  
    current_path = os.path.abspath('.')  
  
    while True:  
        title = f"Select a file from: {current_path} (or cancel)"  
          
        try:  
            items = os.listdir(current_path)  
        except PermissionError:  
            console.print(Panel(f"[bold red]Permission denied: {current_path}[/bold red]"))  
            current_path = os.path.abspath(os.path.join(current_path, '..'))  
            continue  
          
        options = []  
        # Parent directory option  
        parent_path = os.path.abspath(os.path.join(current_path, '..'))  
        if parent_path != current_path:  
            options.append("../ (Parent Directory)")  
  
        # Directories  
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_path, d))])  
        for d in dirs:  
            options.append(f"[DIR] {d}")  
  
        # Files  
        files = sorted([f for f in items if os.path.isfile(os.path.join(current_path, f))])  
        for f in files:  
            if os.path.splitext(f)[1].lower() in supported_extensions:  
                options.append(f)  
  
        options.append("[Cancel]")  
  
        selected, index = pick(options, title, indicator='=>')  
  
        if selected == "[Cancel]" or index is None:  
            return None  
  
        if selected == "../ (Parent Directory)":  
            current_path = parent_path  
            continue  
          
        if selected.startswith("[DIR] "):  
            dir_name = selected[6:]  
            current_path = os.path.join(current_path, dir_name)  
            continue  
  
        # It's a file  
        return os.path.join(current_path, selected)  
  
def open_folder_picker():  
    """Displays an interactive folder picker to browse and select a directory."""  
    current_path = os.path.abspath('.')  
  
    while True:  
        title = f"Select a folder from: {current_path} (or cancel)"  
        try:  
            items = os.listdir(current_path)  
        except PermissionError:  
            console.print(Panel(f"[bold red]Permission denied: {current_path}[/bold red]"))  
            current_path = os.path.abspath(os.path.join(current_path, '..'))  
            continue  
  
        options = []  
        parent_path = os.path.abspath(os.path.join(current_path, '..'))  
        if parent_path != current_path:  
            options.append("../ (Parent Directory)")  
  
        options.append("[Select this folder]")  
  
        dirs = sorted([d for d in items if os.path.isdir(os.path.join(current_path, d))])  
        for d in dirs:  
            options.append(f"[DIR] {d}")  
  
        options.append("[Cancel]")  
  
        selected, index = pick(options, title, indicator='=>')  
  
        if selected == "[Cancel]" or index is None:  
            return None  
  
        if selected == "../ (Parent Directory)":  
            current_path = parent_path  
            continue  
  
        if selected == "[Select this folder]":  
            return current_path  
  
        if selected.startswith("[DIR] "):  
            dir_name = selected[6:]  
            current_path = os.path.join(current_path, dir_name)  
            continue  
  
def resolve_image_model(preferred_model):  
    """Resolve an installed image model, preferring flux2-klein variants."""  
    try:  
        models_info = ollama.list()  
        available_models = [model['model'] for model in models_info.get('models', [])]  
    except Exception as e:  
        console.print(Panel(f"[bold red]Error fetching models: {e}[/bold red]"))  
        return None  
  
    if preferred_model in available_models:  
        return preferred_model  
  
    flux_variants = [m for m in available_models if "flux2-klein" in m]  
    if not flux_variants:  
        return None  
  
    if "x/flux2-klein:9b" in flux_variants:  
        return "x/flux2-klein:9b"  
    if "x/flux2-klein" in flux_variants:  
        return "x/flux2-klein"  
  
    return flux_variants[0]  
  
def get_multiline_input():  
    """Gets multi-line input from the user using prompt_toolkit."""  
    return session.prompt(  
        "> ",  
        placeholder="Enter your prompt. Press [Meta+Enter] or [Esc] then [Enter] to submit.",  
        multiline=True,  
        key_bindings=bindings,  
        prompt_continuation="... "  
    )  
  
from rich.progress_bar import ProgressBar  
  
def save_conversation(messages, model, file_name, is_document_chat):  
    """Saves the conversation to a file."""  
    if not messages:  
        console.print(Panel("[bold yellow]Conversation is empty. Nothing to save.[/bold yellow]"))  
        return  
  
    console.print(Panel("Select format: [1] Markdown (.md) [2] Plain Text (.txt)", title="[bold green]Save Conversation[/bold green]"))  
    format_choice = Prompt.ask("Choose format", choices=["1", "2"], default="1")  
    file_extension = ".md" if format_choice == "1" else ".txt"  
      
    default_filename = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}{file_extension}"  
    filename = Prompt.ask("Enter filename", default=default_filename)  
  
    content = f"# Chat Conversation\n\n"  
    content += f"**Model:** {model}\n"  
    content += f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"  
    if is_document_chat and file_name:  
        content += f"**Document:** {os.path.basename(file_name)}\n"  
    content += "\n---\n\n"  
  
    for msg in messages:  
        role = msg['role'].capitalize()  
        # Sanitize content by removing special characters that might interfere with markdown  
        msg_content = re.sub(r'([<>])', r'\\\1', msg['content'])  
  
        if format_choice == "1": # Markdown  
            if role.lower() == 'user':  
                content += f"### 👤 {role}\n\n{msg_content}\n\n"  
            else:  
                # For assistant, ensure code blocks are correctly formatted  
                # This is a simple heuristic, might need more robust parsing for complex cases  
                if "```" in msg_content:  
                    content += f"### 🤖 {role}\n\n{msg_content}\n\n"  
                else:  
                    content += f"### 🤖 {role}\n\n```\n{msg_content}\n```\n\n"  
        else: # Plain Text  
            content += f"--- {role} ---\n{msg_content}\n\n"  
  
    try:  
        with open(filename, 'w', encoding='utf-8') as f:  
            f.write(content)  
        console.print(Panel(f"Conversation saved to [bold green]{filename}[/bold green]"))  
    except Exception as e:  
        console.print(Panel(f"[bold red]Error saving file: {e}[/bold red]"))  
  
def print_image_help():  
    """Prints help for image generation mode."""  
    table = Table.grid(expand=True, padding=(0, 2))  
    table.add_column(width=24)  
    table.add_column()  
  
    table.add_row(Text("Image Generation", style="bold green"), "")  
    table.add_row(Text("/set width <n>", style="white"), Text("Set image width in pixels", style="white"))  
    table.add_row(Text("/set height <n>", style="white"), Text("Set image height in pixels", style="white"))  
    table.add_row(Text("/set steps <n>", style="white"), Text("Set number of diffusion steps", style="white"))  
    table.add_row(Text("/set seed <n>", style="white"), Text("Set a random seed for reproducibility", style="white"))  
    table.add_row()  
    table.add_row(Text("General", style="bold yellow"), "")  
    table.add_row(Text("/help", style="white"), Text("Display this help message", style="white"))  
    table.add_row(Text("/new", style="white"), Text("Return to the main menu", style="white"))  
    table.add_row(Text("/exit or /quit", style="white"), Text("Exit the application", style="white"))  
  
    console.print(Panel(table, title="[bold]Image Mode Help[/bold]", border_style="white", expand=True))  
  
def render_image_settings(image_state, model_name):  
    steps = image_state.get("steps")  
    seed = image_state.get("seed")  
    steps_text = str(steps) if steps is not None else "default"  
    seed_text = str(seed) if seed is not None else "random"  
    return (  
        f"Model: {model_name}\n"  
        f"Width: {image_state['width']} | Height: {image_state['height']}\n"  
        f"Steps: {steps_text} | Seed: {seed_text}"  
    )  
  
def build_image_options(image_state):  
    options = {  
        "width": image_state["width"],  
        "height": image_state["height"],  
    }  
    if image_state.get("steps") is not None:  
        options["steps"] = image_state["steps"]  
    if image_state.get("seed") is not None:  
        options["seed"] = image_state["seed"]  
    return options  
  
def decode_image_payload(payload):  
    if not payload:  
        return []  
    if isinstance(payload, dict):  
        if payload.get("images"):  
            return payload["images"]  
        data = payload.get("data")  
        if isinstance(data, list):  
            return [item.get("b64_json") for item in data if isinstance(item, dict) and item.get("b64_json")]  
    return []  
  
def generate_images_v1(model_name, prompt, size, options):  
    """Generate images via the OpenAI-compatible endpoint."""  
    payload = {  
        "model": model_name,  
        "prompt": prompt,  
        "size": size,  
        "n": 1,  
    }  
    if options.get("steps") is not None:  
        payload["steps"] = options["steps"]  
    if options.get("seed") is not None:  
        payload["seed"] = options["seed"]  
  
    data = json.dumps(payload).encode("utf-8")  
    req = urllib.request.Request(  
        "http://localhost:11434/v1/images/generations",  
        data=data,  
        headers={"Content-Type": "application/json"},  
        method="POST",  
    )  
  
    with urllib.request.urlopen(req, timeout=120) as resp:  
        body = resp.read().decode("utf-8")  
    return decode_image_payload(json.loads(body))  
  
def generate_images_cli(model_name, prompt, work_dir, options):  
    """Generate images via the Ollama CLI and return saved file paths."""  
    try:  
        args = ["ollama", "run", model_name, "--verbose"]  
        if options.get("width"):  
            args += ["--width", str(options["width"])]  
        if options.get("height"):  
            args += ["--height", str(options["height"])]  
        if options.get("steps") is not None:  
            args += ["--steps", str(options["steps"])]  
        if options.get("seed") is not None:  
            args += ["--seed", str(options["seed"])]  
        args.append(prompt)  
  
        result = subprocess.run(  
            args,  
            text=True,  
            capture_output=True,  
            cwd=work_dir,  
            timeout=300,  
        )  
    except FileNotFoundError:  
        return []  
    except subprocess.TimeoutExpired:  
        return []  
  
    output = (result.stdout or "") + "\n" + (result.stderr or "")  
    paths = []  
    for line in output.splitlines():  
        if "Image saved to:" in line:  
            path = line.split("Image saved to:", 1)[1].strip()  
            if not os.path.isabs(path):  
                path = os.path.join(work_dir, path)  
            paths.append(path)  
    if paths:  
        return paths  
  
    for name in os.listdir(work_dir):  
        if name.lower().endswith(".png"):  
            paths.append(os.path.join(work_dir, name))  
    return paths  
  
def generate_images(model_name, prompt, options):  
    """Generate images with Ollama, trying the images API first."""  
    size = f"{options.get('width', IMAGE_DEFAULT_WIDTH)}x{options.get('height', IMAGE_DEFAULT_HEIGHT)}"  
    try:  
        images = generate_images_v1(model_name, prompt, size, options)  
        if images:  
            return images, []  
    except urllib.error.HTTPError:  
        pass  
    except urllib.error.URLError:  
        pass  
    except Exception:  
        pass  
  
    try:  
        response = ollama.generate(model=model_name, prompt=prompt, options=options, stream=False)  
        images = decode_image_payload(response if isinstance(response, dict) else None)  
        if images:  
            return images, []  
    except json.JSONDecodeError:  
        pass  
    except Exception as e:  
        if "Extra data" not in str(e):  
            raise  
  
    images = []  
    response_stream = ollama.generate(model=model_name, prompt=prompt, options=options, stream=True)  
    for chunk in response_stream:  
        chunk_images = decode_image_payload(chunk)  
        if chunk_images:  
            images.extend(chunk_images)  
    if images:  
        return images, []  
  
    with tempfile.TemporaryDirectory(prefix="contextual_images_") as tmpdir:  
        paths = generate_images_cli(model_name, prompt, tmpdir, options)  
    return [], paths  
  
def launch_external_terminal_image(prompt, model_name, options, save_dir):  
    """Open a new Terminal window to run ollama image generation in the shell."""  
    args = ["ollama", "run", model_name, "--verbose"]  
    if options.get("width"):  
        args += ["--width", str(options["width"])]  
    if options.get("height"):  
        args += ["--height", str(options["height"])]  
    if options.get("steps") is not None:  
        args += ["--steps", str(options["steps"])]  
    if options.get("seed") is not None:  
        args += ["--seed", str(options["seed"])]  
    args.append(prompt)  
  
    cmd = " ".join(shlex.quote(a) for a in args)  
    terminal_cmd = f"cd {shlex.quote(save_dir)}; {cmd}"  
    terminal_cmd = terminal_cmd.replace('"', '\\"')  
  
    try:  
        subprocess.run(  
            ["osascript", "-e", f'tell application "Terminal" to do script "{terminal_cmd}"'],  
            check=False,  
        )  
        return True  
    except Exception:  
        return False  
  
def parse_image_set_command(user_message, image_state):  
    parts = user_message.strip().split()  
    if len(parts) != 3:  
        return False, "Invalid command. Use /set width <n>, /set height <n>, /set steps <n>, or /set seed <n>."  
  
    _, key, value = parts  
    key = key.lower()  
  
    if key not in {"width", "height", "steps", "seed"}:  
        return False, f"Invalid image parameter: {key}."  
  
    if not value.isdigit():  
        return False, "Value must be a non-negative integer."  
  
    num = int(value)  
    if key in {"width", "height", "steps"} and num <= 0:  
        return False, "Value must be greater than zero."  
  
    image_state[key] = num  
    return True, f"Image parameter [bold]{key}[/bold] set to [bold]{num}[/bold]."  
  
def handle_image_generation():  
    """Runs the image generation loop using the x/flux2-klein model."""  
    resolved_model = resolve_image_model(IMAGE_MODEL)  
    if not resolved_model:  
        console.print(Panel("[bold red]No installed flux2-klein image model found. Pull x/flux2-klein:9b (or another variant) and try again.[/bold red]"))  
        return False  
  
    console.print(Panel("Select a folder to save generated images.", title="[bold green]Image Generation[/bold green]"))  
    save_dir = open_folder_picker()  
    if not save_dir:  
        console.print(Panel("[bold yellow]No folder selected. Returning to main menu.[/bold yellow]"))  
        return False  
  
    image_state = {  
        "width": IMAGE_DEFAULT_WIDTH,  
        "height": IMAGE_DEFAULT_HEIGHT,  
        "steps": 4 if "flux2-klein" in resolved_model else None,  
        "seed": None,  
    }  
  
    console.print(Panel(render_image_settings(image_state, resolved_model), title="[bold green]Current Image Settings[/bold green]"))  
    console.print(Panel("Enter an image prompt. Use /set width|height|steps|seed to adjust parameters.", style="bold white"))  
  
    use_external_terminal = True  
  
    while True:  
        console.print(Rule(title="[bold purple]Image Prompt[/bold purple]", style="white"))  
        user_message = get_multiline_input()  
  
        if not user_message.strip():  
            continue  
  
        lower = user_message.lower().strip()  
        if lower in ['/exit', '/quit']:  
            return True  
        if lower == '/new':  
            return False  
        if lower == '/help':  
            print_image_help()  
            continue  
        if lower.startswith('/set '):  
            ok, message = parse_image_set_command(user_message, image_state)  
            if ok:  
                console.print(Panel(message, style="bold yellow"))  
                console.print(Panel(render_image_settings(image_state, resolved_model), title="[bold green]Current Image Settings[/bold green]"))  
            else:  
                console.print(Panel(f"[bold red]{message}[/bold red]"))  
            continue  
  
        options = build_image_options(image_state)  
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
  
        try:  
            if use_external_terminal:  
                existing = {f for f in os.listdir(save_dir) if f.lower().endswith(".png")}  
                launched = launch_external_terminal_image(user_message, resolved_model, options, save_dir)  
                if not launched:  
                    console.print(Panel("[bold red]Failed to launch Terminal for image generation.[/bold red]"))  
                    continue  
  
                console.print(Panel("Terminal opened. Generate the image there. Press Enter here when it finishes to import.", style="bold yellow"))  
                Prompt.ask("")  
  
                new_files = []  
                for name in os.listdir(save_dir):  
                    if not name.lower().endswith(".png"):  
                        continue  
                    if name not in existing:  
                        new_files.append(os.path.join(save_dir, name))  
  
                if not new_files:  
                    console.print(Panel("[bold red]No new images found in the selected folder.[/bold red]"))  
                    continue  
  
                saved_list = "\n".join(f"[white]{path}[/white]" for path in new_files)  
                console.print(Panel(f"Saved {len(new_files)} image(s):\n{saved_list}", title="[bold green]Image Saved[/bold green]"))  
                continue  
  
            with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:  
                response_start_time = time.time()  
                spinner = Spinner("dots", text=Text("Generating image...", style="yellow"))  
                live.update(  
                    Group(  
                        Rule(title=f"[bold green]{resolved_model}[/bold green]", style="white"),  
                        Align.center(spinner),  
                        Rule(title=Text("Time: 0.00s", style="bold yellow"), style="white", align="right"),  
                    ),  
                    refresh=True,  
                )  
                images, file_paths = generate_images(resolved_model, user_message, options)  
  
            saved_files = []  
            if images:  
                for idx, image_b64 in enumerate(images, start=1):  
                    image_bytes = base64.b64decode(image_b64)  
                    filename = f"contextual_{timestamp}_{idx}.png"  
                    output_path = os.path.join(save_dir, filename)  
                    with open(output_path, "wb") as f:  
                        f.write(image_bytes)  
                    saved_files.append(output_path)  
            elif file_paths:  
                for idx, path in enumerate(file_paths, start=1):  
                    if not os.path.exists(path):  
                        continue  
                    filename = f"contextual_{timestamp}_{idx}.png"  
                    output_path = os.path.join(save_dir, filename)  
                    shutil.copy2(path, output_path)  
                    saved_files.append(output_path)  
  
            if not saved_files:  
                console.print(Panel("[bold red]No image data returned. Ensure the model is installed and supports image generation.[/bold red]"))  
                continue  
  
            saved_list = "\n".join(f"[white]{path}[/white]" for path in saved_files)  
            console.print(Panel(f"Saved {len(saved_files)} image(s):\n{saved_list}", title="[bold green]Image Saved[/bold green]"))  
  
        except Exception as e:  
            console.print(Panel(f"[bold red]Image generation failed: {e}[/bold red]"))  
  
def handle_chat_interaction(chat_model, is_document_chat, ollama_parameters, **kwargs):  
    """Handles the main chat loop for both document and general chat modes."""  
    messages = []  
    turns = 0  
    input_tokens = 0  
    output_tokens = 0  
    start_time = time.time()  
    user_system_prompt = None  
    thinking_enabled = False  
  
    # Internal state for the chat session  
    current_mode_is_doc_chat = is_document_chat  
    collection = kwargs.get("collection")  
    embedding_model = kwargs.get("embedding_model")  
    file_path = kwargs.get("file_path")  
    n_results = 5  
  
    if current_mode_is_doc_chat:  
        console.print(Panel(f"Retrieving the top [bold]{n_results}[/bold] relevant chunks for each query. Use [bold]/chunks N[/bold] to change.", style="dim", expand=False))  
  
    while True:  
        console.print(Rule(title="[bold purple]You[/bold purple]", style="white"))  
        user_message = get_multiline_input()  
  
        if not user_message.strip():  
            continue  
  
        if user_message.lower().strip() in ['/exit', '/quit']:  
            return True  
        if user_message.lower().strip() == '/new':  
            return False  
  
        if user_message.lower().strip() == '/prompt':  
            console.print(Panel("Enter your custom system prompt. Press [Meta+Enter] or [Esc] followed by [Enter] to submit.", title="[bold green]Set System Prompt[/bold green]"))  
            user_system_prompt = get_multiline_input()  
            console.print(Panel(f"System prompt set to: \n\n[italic]{user_system_prompt}[/italic]", style="bold yellow"))  
            continue  
  
  
        if user_message.lower().strip() == '/file':  
            console.print(Panel("Please select an embedding model.", title="[bold green]Switching to Document Chat[/bold green]"))  
            embedding_model = select_embedding_model(True)  
            if not embedding_model:  
                console.print(Panel("[bold yellow]Embedding model selection cancelled. Returning to chat.[/bold yellow]"))  
                continue  
  
            console.print(Panel("Please select a new file.", title="[bold green]Switching File[/bold green]"))  
            new_content, new_path, new_collection = select_and_load_file(embedding_model=embedding_model)  
            if new_content:  
                collection, file_path = new_collection, new_path  
                current_mode_is_doc_chat = True  
                messages, turns = [], 0  
                console.print(Panel(f"Switched to document chat with file: [bold]{os.path.basename(file_path)}[/bold]", title="[green]Mode Switched[/green]", expand=False))  
            else:  
                console.print(Panel("[bold yellow]File switch cancelled. Returning to chat.[/bold yellow]"))  
            continue  
  
        if user_message.lower().strip() == '/csv':  
            console.print(Panel("Select a CSV file to clean and load.", title="[bold green]CSV Mode[/bold green]"))  
            csv_path = open_file_picker()  
            if csv_path:  
                try:  
                    cleaned_path = clean_csv(csv_path)  
                      
                    # Define and apply parameters for cleaned CSV  
                    csv_cleaning_params = {  
                        "num_ctx": 8192,  
                        "num_predict": 2048,  
                        "temperature": 0.2,  
                        "top_p": 0.9,  
                        "top_k": 50,  
                        "repeat_penalty": 1.1,  
                        "seed": 42,  
                        "stop": ["<<END>>"],  
                    }  
                    ollama_parameters.update(csv_cleaning_params)  
  
                    # Display confirmation panel  
                    params_text = "\n".join([f"{key}: {value}" for key, value in csv_cleaning_params.items()])  
                    console.print(Panel(params_text, title="[bold yellow]Default Parameters for Cleaned CSV Applied[/bold yellow]", border_style="yellow", style="yellow"))  
                      
                    console.print(Panel("Now, please select an embedding model.", title="[bold green]Switching to Document Chat[/bold green]"))  
                    embedding_model = select_embedding_model(True)  
                    if not embedding_model:  
                        console.print(Panel("[bold yellow]Embedding model selection cancelled. Returning to chat.[/bold yellow]"))  
                        continue  
  
                    new_content, new_path, new_collection = select_and_load_file(file_path=cleaned_path, embedding_model=embedding_model)  
                    if new_content:  
                        collection, file_path = new_collection, new_path  
                        current_mode_is_doc_chat = True  
                        messages, turns = [], 0  
                        console.print(Panel(f"Switched to document chat with file: [bold]{os.path.basename(file_path)}[/bold]", title="[green]Mode Switched[/green]", expand=False))  
                    else:  
                        console.print(Panel("[bold yellow]File switch cancelled. Returning to chat.[/bold yellow]"))  
  
                except Exception as e:  
                    console.print(Panel(f"[bold red]Error processing CSV: {e}[/bold red]"))  
            else:  
                console.print(Panel("[bold yellow]No file selected. Returning to chat.[/bold yellow]"))  
            continue  
  
        if user_message.lower().strip().startswith('/chunks') and current_mode_is_doc_chat:  
            try:  
                parts = user_message.strip().split()  
                if len(parts) > 1 and parts[1].isdigit():  
                    n_results = int(parts[1])  
                    console.print(Panel(f"Number of chunks to retrieve set to [bold]{n_results}[/bold].", style="bold yellow"))  
                else:  
                    console.print(Panel("[bold red]Invalid command. Use /chunks N, where N is a number.[/bold red]"))  
            except Exception as e:  
                console.print(Panel(f"[bold red]Error setting chunks: {e}[/bold red]"))  
            continue  
          
        if user_message.lower().strip().startswith('/set '):  
            parts = user_message.strip().split()  
            if len(parts) == 2:  
                command = parts[1].lower()  
                if command == 'think':  
                    thinking_enabled = True  
                    console.print(Panel("Thinking mode [bold green]enabled[/bold green]. Model will show its thought process if supported.", title="[yellow]Mode Changed[/yellow]"))  
                    continue  
                elif command == 'nothink':  
                    thinking_enabled = False  
                    console.print(Panel("Thinking mode [bold red]disabled[/bold red].", title="[yellow]Mode Changed[/yellow]"))  
                    continue  
  
                preset_name = parts[1]  
                if preset_name in OLLAMA_PARAMETER_PRESETS:  
                    ollama_parameters.update(OLLAMA_PARAMETER_PRESETS[preset_name])  
                    params_text = "\n".join([f"{key}: {value}" for key, value in OLLAMA_PARAMETER_PRESETS[preset_name].items()])  
                    console.print(Panel(params_text, title=f"[bold yellow]'{preset_name.capitalize()}' Preset Applied[/bold yellow]", border_style="yellow", style="yellow"))  
                else:  
                    console.print(Panel(f"[bold red]Invalid preset or command: {preset_name}[/bold red]"))  
            elif len(parts) >= 4 and parts[1] == "parameter":  
                _, _, param_name, param_value = parts  
                if param_name in VALID_OLLAMA_PARAMETERS:  
                    try:  
                        param_type = VALID_OLLAMA_PARAMETERS[param_name]  
                        if param_type == list:  
                            ollama_parameters[param_name] = [param_value]  
                        else:  
                            ollama_parameters[param_name] = param_type(param_value)  
                        console.print(Panel(f"Parameter [bold]{param_name}[/bold] set to [bold]{ollama_parameters[param_name]}[/bold].", style="bold yellow"))  
                    except ValueError:  
                        console.print(Panel(f"[bold red]Invalid value type for {param_name}. Expected {VALID_OLLAMA_PARAMETERS[param_name].__name__}.[/bold red]"))  
                else:  
                    console.print(Panel(f"[bold red]Invalid parameter: {param_name}[/bold red]"))  
            else:  
                console.print(Panel("[bold red]Invalid command. Use /set <preset>, /set think, /set nothink, or /set parameter <name> <value>.[/bold red]"))  
            continue  
  
        if user_message.lower().strip() == '/save':  
            file_name_for_save = os.path.basename(file_path) if current_mode_is_doc_chat and file_path else None  
            save_conversation(messages, chat_model, file_name_for_save, current_mode_is_doc_chat)  
            continue  
  
        if user_message.lower().strip() == '/switch':  
            new_model = select_chat_model(True)  
            if new_model:  
                chat_model = new_model  
                console.print(Panel(f"Switched to model: [bold]{chat_model}[/bold]", title="[green]Model Switched[/green]", expand=False))  
            continue  
        if user_message.lower().strip() == '/help':  
            print_help()  
            continue  
        if user_message.lower().strip() == '/stats':  
            file_name = os.path.basename(file_path) if current_mode_is_doc_chat and file_path else 'N/A'  
            print_stats(start_time, chat_model, file_name, turns, input_tokens, output_tokens, ollama_parameters, current_mode_is_doc_chat)  
            continue  
  
        turns += 1  
          
        try:  
            prompt = user_message  
              
            with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:  
                response_start_time = time.time()  
                spinner = Spinner("dots", text=Text("Working...", style="yellow"))  
                live.update(  
                    Group(  
                        Rule(title=f"[bold green]{chat_model}[/bold green]", style="white"),  
                        Align.center(spinner),  
                        Rule(title=Text("Time: 0.00s", style="bold yellow"), style="white", align="right"),  
                    ),  
                    refresh=True,  
                )  
                if current_mode_is_doc_chat:  
                    live.update(Text("Generating embeddings...", justify="center"), refresh=True)  
                    # EDIT: embeddings API uses input= not prompt=
                    response = embed_text(model=embedding_model, text=user_message)  
                      
                    live.update(Text("Querying documents...", justify="center"), refresh=True)  
                    # EDIT: include metadatas for citation
                    results = collection.query(query_embeddings=[response["embedding"]], n_results=n_results, include=["documents","metadatas"])  
                    docs = results['documents'][0]  
                    metas = results['metadatas'][0]  
                    labeled = []  
                    for d, m in zip(docs, metas):  
                        label = f"[{m.get('source','?')} chunk {m.get('chunk','?')}]"
                        labeled.append(f"{label}\n{d}")  
                    context = "\n\n".join(labeled)  
                    # EDIT: Gemma-friendly guardrails inline
                    guard = ("You are a RAG assistant. Use ONLY the context below. "
                             "Cite the bracketed labels. If the answer is not in the context, say 'Not found in the provided context.'")  
                    prompt = f"{guard}\n\nContext:\n{context}\n\nQuestion: {user_message}\n"  
                  
                messages.append({'role': 'user', 'content': prompt})  
                  
                system_prompt_parts = []  
                if thinking_enabled:  
                    system_prompt_parts.append("Think step-by-step...")  
                if user_system_prompt:  
                    system_prompt_parts.append(user_system_prompt)  
                final_system_prompt = "\n\n".join(system_prompt_parts)  
                  
                final_messages = messages  
                if final_system_prompt:  
                    final_messages = [{'role': 'system', 'content': final_system_prompt}] + messages  
  
                spinner = Spinner("dots", text=Text("Waiting for model response...", style="yellow"))  
                live.update(  
                    Group(  
                        Rule(title=f"[bold green]{chat_model}[/bold green]", style="white"),  
                        Align.center(spinner),  
                        Rule(title=Text("Time: 0.00s", style="bold yellow"), style="white", align="right"),  
                    ),  
                    refresh=True,  
                )  
  
                response_stream = ollama.chat(model=chat_model, messages=final_messages, options=ollama_parameters, stream=True)  
                  
                full_response_content = ""  
                thinking_text = ""  
                final_answer = ""  
                full_response_data = {}  
  
                for chunk in response_stream:  
                    chunk_content = chunk['message']['content']  
                    full_response_content += chunk_content  
                      
                    thinking_match = re.search(r"<thinking>(.*?)</thinking>", full_response_content, re.DOTALL)  
                    if thinking_match:  
                        thinking_text = thinking_match.group(1)  
                        final_answer = full_response_content.replace(thinking_match.group(0), "")  
                    else:  
                        final_answer = full_response_content  
  
                    render_items = []  
                    if thinking_text:  
                        render_items.append(Text("Thinking...", style="italic cyan"))  
                        render_items.append(Text(thinking_text, style="cyan"))  
                        render_items.append(Rule(style="cyan"))  
                      
                    render_items.append(Markdown(final_answer))  
                      
                    elapsed_time = time.time() - response_start_time  
                    bot_response_group = Group(  
                        Rule(title=f"[bold green]{chat_model}[/bold green]", style="white"),  
                        *render_items,  
                        Rule(title=Text(f"Time: {elapsed_time:.2f}s", style="bold yellow"), style="white", align="right")  
                    )  
                    live.update(bot_response_group, refresh=True)  
                      
                    if chunk.get('done'):  
                        full_response_data = chunk  
  
                messages.append({'role': 'assistant', 'content': final_answer})  
                if 'prompt_eval_count' in full_response_data:  
                    input_tokens += full_response_data.get('prompt_eval_count', 0)  
                if 'eval_count' in full_response_data:  
                    output_tokens += full_response_data.get('eval_count', 0)  
  
        except KeyboardInterrupt:  
            console.print("\n[bold yellow]Interrupted by user. Returning to prompt.[/bold yellow]")  
            if messages and messages[-1]['role'] == 'user':  
                messages.pop()  
        except Exception as e:  
            console.print(Panel(f"[bold red]An error occurred with Ollama: {e}[/bold red]"))  
  
  
def clean_csv(input_file):  
    """  
    Cleans a CSV file according to the specified steps.  
    """  
    import pathlib  
    fname = pathlib.Path(input_file)  
  
    # Step 1: Encoding  
    try:  
        df = pd.read_csv(fname, encoding='utf-8', on_bad_lines='warn')  
    except UnicodeDecodeError:  
        df = pd.read_csv(fname, encoding='utf-8', errors='replace')  
  
    # Step 2: Single line per turn  
    if 'text' in df.columns:  
        df['text'] = df['text'].str.replace(r'\r?\n+', ' ', regex=True)  
  
    # Step 3: Strip leading/trailing whitespace  
    if 'text' in df.columns:  
        df['text'] = df['text'].str.strip()  
  
    # Step 4: Drop empty or system rows  
    if 'text' in df.columns:  
        df = df[~df['text'].str.match(r'^\s*$|^System:.*', case=False)]  
  
    # Step 5: Unify speaker labels  
    if 'speaker' in df.columns:  
        df['speaker'] = df['speaker'].str.lower().map({  
            'human': 'user',  
            'assistant': 'ai'  
        }).fillna(df['speaker'])  
  
    # Step 6: Handle conversation_id  
    if 'conversation_id' not in df.columns:  
        if 'turn_id' in df.columns:  
            # A new conversation starts when turn_id is less than the previous one (e.g., resets to 0)  
            # This assumes the CSV is chronologically ordered.  
            new_convo_starts = df['turn_id'].diff() < 0  
            convo_group = new_convo_starts.cumsum()  
            df['conversation_id'] = f'conv_{fname.stem}_' + convo_group.astype(str)  
        else:  
            # Fallback if no turn_id, assign one ID for the whole file.  
            df['conversation_id'] = 'conv_' + fname.stem  
  
    # Step 7: Ensure strict integer order within each conversation  
    if 'turn_id' in df.columns and 'conversation_id' in df.columns:  
        df = df.sort_values(['conversation_id', 'turn_id']).reset_index(drop=True)  
  
    # Step 8: Re-export to clean CSV  
    output_path = 'clean_' + fname.name  
    df.to_csv(output_path, index=False, encoding='utf-8')  
    return output_path  
  
  
def main():  
    """Main function to run the RAG chatbot."""  
    parser = argparse.ArgumentParser(description="A RAG chatbot that answers questions based on a provided document.")  
    parser.add_argument("file_path", nargs='?', default=None, help="The absolute path to the document file.")  
    parser.add_argument("-q", "--question", help="A question to ask the chatbot non-interactively.")  
    args = parser.parse_args()  
  
    is_interactive = sys.stdout.isatty() and not args.question  
    ollama_parameters = {}  
  
    while True:  
        print_banner()  
        start_time = time.time()  
          
        # Mode selection  
        mode = "Chat with a document" if args.file_path or args.question else "General chat"  
        if is_interactive and not args.file_path and not args.question:  
            mode_list = """[yellow]1[/yellow]: General chat  
[yellow]2[/yellow]: Image generation  
[yellow]3[/yellow]: Chat with a document  
[yellow]4[/yellow]: Clean CSV  
[yellow]5[/yellow]: Read the README"""  
            console.print(Panel(mode_list, title="[bold green]Choose a Mode[/bold green]", expand=True))  
            mode_choice = IntPrompt.ask("Select a mode", choices=["1", "2", "3", "4", "5"], default=1)  
            if mode_choice == 1:  
                mode = "General chat"  
            elif mode_choice == 2:  
                mode = "Image generation"  
            elif mode_choice == 3:  
                mode = "Chat with a document"  
            elif mode_choice == 4:  
                mode = "Clean CSV"  
            else:  
                mode = "Read README"  
  
        if mode == "Read README":  
            try:  
                script_dir = os.path.dirname(os.path.abspath(__file__))  
                readme_path = os.path.join(script_dir, "README.md")  
                with open(readme_path, "r", encoding="utf-8") as f:  
                    readme_content = f.read()  
                console.print(Panel(Markdown(readme_content), title="[bold green]README.md[/bold green]", border_style="white", expand=True))  
                console.print(Panel("Press Enter to return to the main menu.", style="bold yellow"))  
                Prompt.ask()  
                continue  
            except FileNotFoundError:  
                console.print(Panel("[bold red]README.md not found.[/bold red]"))  
                continue  
  
        if mode == "Image generation":  
            should_exit = handle_image_generation()  
            if should_exit:  
                break  
            continue  
  
        if mode == "Clean CSV":  
            console.print(Panel("Select a CSV file to clean.", title="[bold green]Clean CSV Mode[/bold green]"))  
            csv_path = open_file_picker()  
            if csv_path:  
                try:  
                    with Live(console=console, auto_refresh=False, vertical_overflow="visible") as live:  
                        start_clean = time.time()  
                        spinner = Spinner("dots", text=Text("Cleaning CSV...", style="yellow"))  
                        live.update(  
                            Group(  
                                Rule(title="[bold green]Clean CSV[/bold green]", style="white"),  
                                Align.center(spinner),  
                                Rule(title=Text("Time: 0.00s", style="bold yellow"), style="white", align="right"),  
                            ),  
                            refresh=True,  
                        )  
                        cleaned_path = clean_csv(csv_path)  
                      
                    # Define and apply parameters for cleaned CSV  
                    csv_cleaning_params = {  
                        "num_ctx": 8192,  
                        "num_predict": 2048,  
                        "temperature": 0.2,  
                        "top_p": 0.9,  
                        "top_k": 50,  
                        "repeat_penalty": 1.1,  
                        "seed": 42,  
                        "stop": ["<<END>>"],  
                    }  
                    ollama_parameters.update(csv_cleaning_params)  
  
                    # Display confirmation panel  
                    params_text = "\n".join([f"{key}: {value}" for key, value in csv_cleaning_params.items()])  
                    console.print(Panel(params_text, title="[bold yellow]Default Parameters for Cleaned CSV Applied[/bold yellow]", border_style="yellow", style="yellow"))  
  
                    console.print(Panel(f"Successfully cleaned CSV and saved to [bold green]{cleaned_path}[/bold green].\nNow, select a file to chat with.", title="[green]Cleaning Complete[/green]"))  
                    mode = "Chat with a document" # Transition to chat with doc mode  
                except Exception as e:  
                    console.print(Panel(f"[bold red]Error cleaning CSV: {e}[/bold red]"))  
                    continue  
            else:  
                console.print(Panel("[bold yellow]No file selected. Returning to main menu.[/bold yellow]"))  
                continue  
  
        if mode == "Chat with a document":  
            embedding_model = select_embedding_model(is_interactive)  
            if not embedding_model:  
                break  
              
            chat_model = select_chat_model(is_interactive)  
            if not chat_model:  
                break  
  
            file_content, file_path, collection = select_and_load_file(args.file_path, embedding_model)  
            if not file_content:  
                if not is_interactive:  
                    break  
                else:  
                    continue  
  
            console.print(Panel(f"Using embedding model: [bold]{embedding_model}[/bold] and chat model: [bold]{chat_model}[/bold]", title="[green]Ready[/green]", expand=False))  
  
            if args.question:  
                # Non-interactive document question  
                # EDIT: embeddings API uses input= not prompt=
                response = embed_text(model=embedding_model, text=args.question)  
                # EDIT: include metadatas for citation
                results = collection.query(query_embeddings=[response["embedding"]], n_results=5, include=["documents","metadatas"])  
                docs = results['documents'][0]  
                metas = results['metadatas'][0]  
                labeled = []  
                for d, m in zip(docs, metas):  
                    label = f"[{m.get('source','?')} chunk {m.get('chunk','?')}]"
                    labeled.append(f"{label}\n{d}")  
                context = "\n\n".join(labeled)  
                # EDIT: Gemma-friendly guardrails inline
                guard = ("You are a RAG assistant. Use ONLY the context below. "
                         "Cite the bracketed labels. If the answer is not in the context, say 'Not found in the provided context.'")  
                prompt = f"{guard}\n\nContext:\n{context}\n\nQuestion: {args.question}\n"  
                  
                response = ollama.chat(model=chat_model, messages=[{'role': 'user', 'content': prompt}], options=ollama_parameters)  
                console.print(Panel(Markdown(response['message']['content']), title=f"[bold green]{chat_model}[/bold green]"))  
                break  
            else:  
                should_exit = handle_chat_interaction(  
                    chat_model=chat_model,  
                    is_document_chat=True,  
                    ollama_parameters=ollama_parameters,  
                    collection=collection,  
                    embedding_model=embedding_model,  
                    file_path=file_path  
                )  
                if should_exit:  
                    break  
  
        elif mode == "General chat":  
            chat_model = select_chat_model(is_interactive)  
            if not chat_model:  
                break  
              
            console.print(Panel(f"Using model: [bold]{chat_model}[/bold]", title="[green]Ready for General Chat[/green]", expand=False))  
            should_exit = handle_chat_interaction(chat_model=chat_model, is_document_chat=False, ollama_parameters=ollama_parameters)  
            if should_exit:  
                break  
          
        end_time = time.time()  
        console.print(Panel(f"Session lasted: {end_time - start_time:.2f} seconds", style="bold yellow"))  
          
        if not is_interactive:  
            break  
  
  
  
if __name__ == "__main__":  
    main()
