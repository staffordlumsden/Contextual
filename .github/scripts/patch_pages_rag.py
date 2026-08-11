from pathlib import Path
import re

path = Path("index.html")
text = path.read_text(encoding="utf-8")

old_css = ".pipeline{margin-top:26px;border:1px solid var(--line);background:#030a05;padding:18px;overflow:auto}.pipeline pre{min-width:760px;margin:0;color:#99d7a3;font:12px/1.35 var(--mono)}"
new_css = ".pipeline{margin-top:26px;border:1px solid var(--line);background:#030a05;padding:22px 18px;overflow:hidden}.pipeline pre{width:max-content;max-width:100%;margin:0 auto;color:#99d7a3;font:clamp(9px,1.05vw,12px)/1.28 var(--mono);white-space:pre}.pipeline-note{margin:14px auto 0;max-width:680px;text-align:center;color:var(--muted);font-size:11px}.pipeline-note b{color:var(--cyan)}"

new_diagram = """┌─────────────────────┐
│   SOURCE MATERIAL    │
│ PDF  DOCX  TXT  MD   │
│ CSV  XLSX            │
└──────────┬──────────┘
           │
           ▼
    ┌──────────────┐
    │   CHONKIE    │
    │ chunk+overlap│
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  DOC EMBED   │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐      ┌──────────────┐
    │   CHROMADB   │ ◄────│ QUERY EMBED  │ ◄──── QUESTION
    │ vector store │      └──────────────┘
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │ TOP K CHUNKS │
    └──────┬───────┘
           │
           ▼
    ┌──────────────┐
    │  LOCAL LLM   │
    └──────┬───────┘
           │
           ▼
         ANSWER"""

if old_css not in text:
    raise SystemExit("Expected pipeline CSS not found; refusing broad rewrite")
text = text.replace(old_css, new_css, 1)

pattern = re.compile(
    r'(<div class="pipeline" aria-label="Document RAG pipeline"><pre>\n).*?(\n</pre></div>)',
    re.S,
)
replacement = (
    r"\1"
    + new_diagram
    + '\n</pre><div class="pipeline-note"><b>INDEX PATH ↓</b> documents are chunked and stored once &nbsp; // &nbsp; <b>QUERY PATH ←</b> each question enters through its own embedding</div></div>'
)
text, count = pattern.subn(replacement, text, count=1)
if count != 1:
    raise SystemExit(f"Expected exactly one RAG pipeline block; found {count}")

mobile_marker = "@media(max-width:620px){.wrap{"
mobile_replacement = "@media(max-width:620px){.pipeline{overflow-x:auto}.pipeline pre{min-width:520px;font-size:8px}.wrap{"
if mobile_marker not in text:
    raise SystemExit("Expected mobile media query not found")
text = text.replace(mobile_marker, mobile_replacement, 1)

path.write_text(text, encoding="utf-8")
