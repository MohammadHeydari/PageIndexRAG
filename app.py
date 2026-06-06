"""
Streamlit UI for DIY PageIndex RAG + Elasticsearch

Run:
    streamlit run app.py
"""

import os, re, json, asyncio, tempfile
import streamlit as st
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

def get_all_nodes(doc_id: str) -> list[dict]:
    result = es.search(
        index=ES_INDEX,
        body={"query": {"term": {"doc_id": doc_id}}, "size": 200},
    )
    return [h["_source"] for h in result["hits"]["hits"]]

# --- PDF ---
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

async def hybrid_search(doc_id: str, query: str) -> tuple[list[dict], list[str], str]:
    candidates = es_search(doc_id, query, top_k=10)
    if not candidates:
        return [], [], "No candidates found by Elasticsearch."

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

    raw = await call_llm(prompt)
    result = parse_json_response(raw)
    return candidates, result.get("node_list", []), result.get("thinking", "")

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

# --- Streamlit UI ---
st.set_page_config(page_title="PageIndex RAG", page_icon="📄", layout="wide")
st.title("DIY PageIndex RAG")
st.caption("Reasoning-first document QA — powered by Elasticsearch + LLM")

# Sidebar
with st.sidebar:
    st.header("Document")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"])
    st.divider()
    st.header("Status")
    es_ok = es.ping()
    if es_ok:
        st.success("Elasticsearch connected")
    else:
        st.error("Elasticsearch not reachable")

# Main area
if not es_ok:
    st.warning("Start Elasticsearch and refresh the page.")
    st.stop()

create_index_if_not_exists()

if uploaded_file is None:
    st.info("Upload a PDF from the sidebar to get started.")
    st.stop()

# Save uploaded file
pdf_path = os.path.join(DOWNLOAD_DIR, uploaded_file.name)
with open(pdf_path, "wb") as f:
    f.write(uploaded_file.getbuffer())

doc_id = os.path.splitext(uploaded_file.name)[0]

# Index document
if not doc_already_indexed(doc_id):
    with st.spinner("Parsing and indexing document..."):
        sections = extract_structure(pdf_path)
        sections = asyncio.run(build_summaries(sections))
        index_sections(doc_id, sections)
    st.success(f"Indexed {len(sections)} sections")
else:
    st.success("Document already indexed")

# Show document tree
with st.expander("Document Tree", expanded=False):
    nodes = get_all_nodes(doc_id)
    nodes_sorted = sorted(nodes, key=lambda x: x.get("page", 0))
    for n in nodes_sorted:
        indent = "   " if n.get("level", 1) == 2 else ""
        st.markdown(f"{indent}**[{n['node_id']}]** p.{n['page']} — {n['title']}")
        st.caption(f"{indent}{n.get('summary', '')}")

st.divider()

# Query
query = st.text_input("Ask a question about the document", placeholder="What are the main conclusions?")

if st.button("Search", disabled=not query):
    with st.spinner("Searching..."):
        candidates, selected_ids, reasoning = asyncio.run(hybrid_search(doc_id, query))

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("ES Candidates")
        for c in candidates:
            marker = "✓" if c["node_id"] in selected_ids else " "
            st.markdown(f"`{marker}` **[{c['node_id']}]** p.{c['page']} — {c['title']}")

    with col2:
        st.subheader("LLM Reasoning")
        st.write(reasoning)

    st.divider()
    st.subheader("Answer")
    with st.spinner("Generating answer..."):
        answer = asyncio.run(generate_answer(doc_id, selected_ids, query))
    st.write(answer)

    with st.expander("Selected nodes detail"):
        for nid in selected_ids:
            result = es.get(index=ES_INDEX, id=f"{doc_id}_{nid}", ignore=404)
            if result.get("found"):
                src = result["_source"]
                st.markdown(f"**[{nid}] {src['title']}** — page {src['page']}")
                st.text(src["text"][:600] + "...")