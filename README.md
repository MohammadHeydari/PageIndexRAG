# DIY PageIndex-style Vectorless RAG

A lightweight, reasoning-first RAG pipeline inspired by [PageIndex](https://pageindex.ai) — built from scratch with **no vector database**, **no chunking**, and **no external retrieval API**.

Instead of splitting documents into fixed chunks and searching an embedding space, this project:
1. Parses a PDF into a **hierarchical section tree**
2. Uses an **LLM to reason over the tree** and select relevant nodes
3. Extracts text from those nodes and generates a **grounded, traceable answer**

---

## How It Works

```
PDF
 └─► pymupdf extracts headings + text
        └─► LLM summarizes each section (one sentence)
               └─► LLM reasons over the tree → picks relevant node IDs
                      └─► Text from selected nodes → LLM generates final answer
```

No embeddings. No cosine similarity. No top-K lookup.
Just structured document understanding + LLM reasoning.

---

## Quickstart

**1. Clone the repo**
```bash
git clone https://github.com/MohammadHeydari/PageIndexRAG.git
cd diy-pageindex-rag
```

**2. Install dependencies**
```bash
pip install pymupdf openai requests
```

**3. Set your API key**
```bash
export GAPGPT_API_KEY="your_key_here"
```

**4. Run**
```bash
python pageindex_.py
```

---

## Configuration

Open `pageindex_.py` and edit the config section at the top:

```python
GAPGPT_API_KEY  = os.getenv("GAPGPT_API_KEY", "your_key_here")
GAPGPT_MODEL    = "gapgpt-qwen-3.5"   # or: gapgpt-gpt-4o

PDF_URL = "https://arxiv.org/pdf/2403.06023.pdf"  # or a local path: "./my_doc.pdf"
QUERY   = "What are the main conclusions of this document?"
```

This project uses [GapGPT](https://gapgpt.app) as the LLM provider (OpenAI-compatible API).
You can swap it for any OpenAI-compatible endpoint by changing `GAPGPT_BASE_URL` and `GAPGPT_MODEL`.

---

## Example Output

```
============================================================
  DIY PageIndex-style Vectorless RAG
============================================================
Extracting Document Structure
   20 Sections Found

Creating Summary for 20 Sections ...

Searching ...

LLM Reasoning:
The question asks for the main conclusions. Node [node_014] is
explicitly titled 'Conclusion'. Node [node_013] 'Results and
Discussion' contains the key quantitative findings.

Selected Nodes: ['node_013', 'node_014']

════════════════════════════════════════════════════════════
Final Response
════════════════════════════════════════════════════════════
1. PSC method improved deep learning classifier performance.
2. Best accuracy: 81.91% (FastText + LSTM + PSC).
3. BERT underperformed due to domain mismatch with informal text.

Used nodes: node_013, node_014
════════════════════════════════════════════════════════════

Results stored in rag_result.json
```

---

## Output File

Results are saved to `rag_result.json`:

```json
{
  "pdf": "https://...",
  "query": "What are the main conclusions?",
  "selected_nodes": ["node_013", "node_014"],
  "answer": "...",
  "tree_summary": [
    { "id": "node_013", "title": "Results and Discussion", "page": 5, "summary": "..." },
    { "id": "node_014", "title": "Conclusion", "page": 6, "summary": "..." }
  ]
}
```

---

## Why Not Vector Search?

```
| | Vector RAG | This project |
|---|---|---|
| Retrieval method | Embedding similarity | LLM reasoning over tree |
| Document structure | Ignored (flat chunks) | Preserved (hierarchy) |
| Traceability | Chunk indices | Named node IDs + reasoning |
| External services | Vector DB required | None |
| Best for | Short, uniform docs | Long, structured documents |
```
---

## Requirements

- Python 3.10+
- `pymupdf` — PDF parsing
- `openai` — LLM calls (OpenAI-compatible)
- `requests` — PDF download

---

## Inspired By

- [PageIndex](https://pageindex.ai) — reasoning-first RAG with hierarchical document trees
- [PageIndex Cookbook](https://docs.pageindex.ai) — official examples
