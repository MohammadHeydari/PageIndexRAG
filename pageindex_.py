"""
DIY PageIndex-style Vectorless RAG + Elasticsearch Hybrid Search

Install:
    pip install pymupdf openai requests elasticsearch python-dotenv

Create a .env file:
    GAPGPT_API_KEY=your_key
    ES_PASSWORD=your_elastic_password
"""

import os, re, json, asyncio, requests
import fitz
from openai import AsyncOpenAI
from elasticsearch import Elasticsearch
from dotenv import load_dotenv

load_dotenv()

# --- CONFIG ---
GAPGPT_API_KEY  = os.getenv("GAPGPT_API_KEY", "")
GAPGPT_BASE_URL = "https://api.gapgpt.app/v1"
GAPGPT_MODEL    = "gapgpt-qwen-3.5"

ES_HOST     = "https://localhost:9200"
ES_USER     = "elastic"
ES_PASSWORD = os.getenv("ES_PASSWORD", "")
ES_INDEX    = "rag_nodes"

PDF_URL      = "https://arxiv.org/pdf/2403.06023.pdf"
QUERY        = "What are the main conclusions of this document?"
DOWNLOAD_DIR = "./data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# --- Clients ---
llm = AsyncOpenAI(api_key=GAPGPT_API_KEY, base_url=GAPGPT_BASE_URL)

es = Elasticsearch(
    ES_HOST,
    basic_auth=(ES_USER, ES_PASSWORD),
    verify_certs=False,
    ssl_show_warn=False,
)

# --- LLM ---
async def call_llm(prompt: str, temperature: float = 0.0) -> str:
    resp = await llm.chat.completions.create(
        model=GAPGPT_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=2000,
    )
    return resp.choices[0].message.content.strip()

def parse_json_response(raw: str) -> dict:
    clean = raw.strip()
    clean = re.sub(r"^```(?:json)?", "", clean).rstrip("```").strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', clean, re.DOTALL)
        if m:
            return json.loads(m.group())
        raise

# --- Elasticsearch ---
def create_index_if_not_exists():
    if not es.indices.exists(index=ES_INDEX):
        es.indices.create(
            index=ES_INDEX,
            body={
                "mappings": {
                    "properties": {
                        "doc_id":  {"type": "keyword"},
                        "node_id": {"type": "keyword"},
                        "title":   {"type": "text"},
                        "summary": {"type": "text"},
                        "text":    {"type": "text"},
                        "level":   {"type": "integer"},
                        "page":    {"type": "integer"},
                    }
                }
            }
        )
        print("Index created")
    else:
        print("Index already exists")

def index_sections(doc_id: str, sections: list[dict]):
    ops = []
    for s in sections:
        ops.append({"index": {"_index": ES_INDEX, "_id": f"{doc_id}_{s['id']}"}})
        ops.append({
            "doc_id":  doc_id,
            "node_id": s["id"],
            "title":   s["title"],
            "summary": s.get("summary", ""),
            "text":    s["text"],
            "level":   s["level"],
            "page":    s["page"],
        })
    es.bulk(body=ops)
    es.indices.refresh(index=ES_INDEX)
    print(f"{len(sections)} nodes indexed in Elasticsearch")

def doc_already_indexed(doc_id: str) -> bool:
    result = es.count(
        index=ES_INDEX,
        body={"query": {"term": {"doc_id": doc_id}}}
    )
    return result["count"] > 0

def es_search(doc_id: str, query: str, top_k: int = 10) -> list[dict]:
    result = es.search(
        index=ES_INDEX,
        body={
            "query": {
                "bool": {
                    "must": [
                        {"term": {"doc_id": doc_id}},
                        {
                            "multi_match": {
                                "query": query,
                                "fields": ["title^3", "summary^2", "text"],
                            }
                        },
                    ]
                }
            },
            "size": top_k,
        }
    )
    return [hit["_source"] for hit in result["hits"]["hits"]]

# --- PDF ---
def download_pdf(url: str) -> str:
    fname = url.split("/")[-1]
    if not fname.endswith(".pdf"):
        fname += ".pdf"
    path = os.path.join(DOWNLOAD_DIR, fname)
    if not os.path.exists(path):
        print(f"Downloading: {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f"Saved: {path}")
    else:
        print(f"File already exists: {path}")
    return path

def extract_structure(pdf_path: str) -> list[dict]:
    doc = fitz.open(pdf_path)
    sections, current_section = [], None
    node_counter = [0]

    def new_id():
        node_counter[0] += 1
        return f"node_{node_counter[0]:03d}"

    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            if block["type"] != 0:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue
                max_size = max(s["size"] for s in spans)
                line_text = " ".join(s["text"] for s in spans).strip()
                if not line_text:
                    continue
                is_bold = any(s["flags"] & 2**4 for s in spans)
                is_heading = (
                    (max_size >= 13 and len(line_text) < 120)
                    or (is_bold and max_size >= 11 and len(line_text) < 80)
                )
                if is_heading:
                    current_section = {
                        "id": new_id(), "title": line_text,
                        "level": 1 if max_size >= 15 else 2,
                        "page": page_num, "text": "",
                    }
                    sections.append(current_section)
                else:
                    if current_section is None:
                        current_section = {
                            "id": new_id(), "title": "Preamble",
                            "level": 1, "page": page_num, "text": "",
                        }
                        sections.append(current_section)
                    current_section["text"] += line_text + " "
    doc.close()
    return [s for s in sections if len(s["text"].strip()) > 30 or s["level"] == 1]

async def build_summaries(sections: list[dict]) -> list[dict]:
    print(f"Building summaries for {len(sections)} sections...")
    section_list = "\n".join(
        f"[{s['id']}] {s['title']}\n{s['text'][:400]}\n" for s in sections
    )
    prompt = f"""For each section below, write a ONE-sentence summary (max 20 words).
Reply ONLY with valid JSON, no markdown:
{{"summaries": {{"node_id": "summary", ...}}}}

Sections:
{section_list}"""
    raw = await call_llm(prompt)
    summaries = parse_json_response(raw).get("summaries", {})
    for s in sections:
        s["summary"] = summaries.get(s["id"], s["title"])
    return sections

# --- Hybrid Search ---
async def hybrid_search(doc_id: str, query: str) -> list[str]:
    print("ES: fast keyword search...")
    candidates = es_search(doc_id, query, top_k=10)
    print(f"{len(candidates)} candidates found")

    if not candidates:
        print("ES returned no results")
        return []

    tree_repr = "\n".join(
        f"[{c['node_id']}] (p.{c['page']}) {c['title']}\n    -> {c['summary']}"
        for c in candidates
    )

    prompt = f"""You are selecting the most relevant sections to answer a question.
Each candidate shows: [id] (page) Title -> summary

Question: {query}

Candidates (pre-filtered by keyword search):
{tree_repr}

Reply ONLY with valid JSON, no markdown:
{{
  "thinking": "brief reasoning",
  "node_list": ["node_id_1", "node_id_2"]
}}"""

    print("LLM reasoning over candidates...")
    raw = await call_llm(prompt)
    result = parse_json_response(raw)

    print("\nLLM Reasoning:")
    print(result.get("thinking", "")[:500])
    print(f"\nSelected nodes: {result.get('node_list')}")
    return result.get("node_list", [])

async def generate_answer(doc_id: str, node_ids: list[str], query: str) -> str:
    if not node_ids:
        return "No relevant sections found."

    context_parts = []
    for nid in node_ids:
        result = es.get(index=ES_INDEX, id=f"{doc_id}_{nid}", ignore=404)
        if not result.get("found"):
            continue
        src = result["_source"]
        context_parts.append(
            f"=== [{nid}] {src['title']} (page {src['page']}) ===\n{src['text'].strip()}"
        )

    combined = "\n\n".join(context_parts) or "No context."

    prompt = f"""Answer the question using ONLY the provided context.
Be concise and accurate. Mention node IDs used at the end.

Question: {query}

Context:
{combined}

Answer:"""

    return await call_llm(prompt, temperature=0.1)

# --- Main ---
async def main():
    print("=" * 60)
    print("  DIY PageIndex RAG + Elasticsearch Hybrid Search")
    print("=" * 60)

    if not es.ping():
        raise ConnectionError("Cannot connect to Elasticsearch")
    print("Elasticsearch connected")

    create_index_if_not_exists()

    pdf_path = download_pdf(PDF_URL)
    doc_id = os.path.splitext(os.path.basename(pdf_path))[0]

    if doc_already_indexed(doc_id):
        print(f"Document '{doc_id}' already indexed, skipping ingestion")
    else:
        print("Extracting document structure...")
        sections = extract_structure(pdf_path)
        print(f"{len(sections)} sections found")
        sections = await build_summaries(sections)
        index_sections(doc_id, sections)

    selected_ids = await hybrid_search(doc_id, QUERY)

    print("\nGenerating final answer...")
    answer = await generate_answer(doc_id, selected_ids, QUERY)

    print("\n" + "=" * 60)
    print("Final Answer")
    print("=" * 60)
    print(answer)
    print("=" * 60)

    with open("rag_result.json", "w", encoding="utf-8") as f:
        json.dump({
            "pdf": PDF_URL,
            "query": QUERY,
            "selected_nodes": selected_ids,
            "answer": answer,
        }, f, ensure_ascii=False, indent=2)
    print("\nResults saved to rag_result.json")


if __name__ == "__main__":
    asyncio.run(main())