import os, re, json, asyncio, requests
import fitz  # pymupdf
from openai import AsyncOpenAI


# CONFIG

GAPGPT_API_KEY  = os.getenv("GAPGPT_API_KEY", "Your GAPGPT_API_KEY")
GAPGPT_BASE_URL = "https://api.gapgpt.app/v1"
GAPGPT_MODEL    = "gapgpt-qwen-3.5"

PDF_URL      = "https://arxiv.org/pdf/2403.06023.pdf"
QUERY        = "What are the main conclusions of this document?"
DOWNLOAD_DIR = "./data"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

# LLM helper
llm = AsyncOpenAI(api_key=GAPGPT_API_KEY, base_url=GAPGPT_BASE_URL)

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



def download_pdf(url: str) -> str:
    fname = url.split("/")[-1]
    if not fname.endswith(".pdf"):
        fname += ".pdf"
    path = os.path.join(DOWNLOAD_DIR, fname)
    if not os.path.exists(path):
        print(f"Download: {url}")
        r = requests.get(url, timeout=60)
        r.raise_for_status()
        with open(path, "wb") as f:
            f.write(r.content)
        print(f" Store: {path}")
    else:
        print(f" Fail Available:  {path}")
    return path


def extract_structure(pdf_path: str) -> list[dict]:

    doc = fitz.open(pdf_path)
    sections = []
    current_section = None
    node_counter = [0]

    def new_id():
        node_counter[0] += 1
        return f"node_{node_counter[0]:03d}"

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
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


                is_bold = any(s["flags"] & 2**4 for s in spans)  # bold flag
                is_heading = (
                    (max_size >= 13 and len(line_text) < 120)
                    or (is_bold and max_size >= 11 and len(line_text) < 80)
                )

                if is_heading:

                    current_section = {
                        "id":    new_id(),
                        "title": line_text,
                        "level": 1 if max_size >= 15 else 2,
                        "page":  page_num,
                        "text":  "",
                    }
                    sections.append(current_section)
                else:

                    if current_section is None:
                        current_section = {
                            "id":    new_id(),
                            "title": "Introduction / Preamble",
                            "level": 1,
                            "page":  page_num,
                            "text":  "",
                        }
                        sections.append(current_section)
                    current_section["text"] += line_text + " "

    doc.close()


    sections = [s for s in sections if len(s["text"].strip()) > 30 or s["level"] == 1]
    return sections


async def build_summaries(sections: list[dict]) -> list[dict]:

    print(f"\nCreating Summary for {len(sections)}  Section ...")

    # batch
    section_list = "\n".join(
        f"[{s['id']}] {s['title']}\n{s['text'][:400]}\n"
        for s in sections
    )

    prompt = f"""For each section below, write a ONE-sentence summary (max 20 words).
Reply ONLY with valid JSON, no markdown:
{{
  "summaries": {{
    "node_id": "one sentence summary",
    ...
  }}
}}

Sections:
{section_list}"""

    raw = await call_llm(prompt)
    data = parse_json_response(raw)
    summaries: dict = data.get("summaries", {})


    for s in sections:
        s["summary"] = summaries.get(s["id"], s["title"])

    return sections


async def search_tree(sections: list[dict], query: str) -> list[str]:

    tree_repr = "\n".join(
        f"  {'  ' * (s['level']-1)}[{s['id']}] (p.{s['page']}) {s['title']}\n"
        f"  {'  ' * (s['level']-1)}    → {s['summary']}"
        for s in sections
    )

    prompt = f"""You are searching a document tree to find sections relevant to a question.
Each node shows: [id] (page) Title → summary

Question: {query}

Document tree:
{tree_repr}

Reply ONLY with valid JSON, no markdown:
{{
  "thinking": "brief reasoning about which sections are relevant",
  "node_list": ["node_id_1", "node_id_2"]
}}"""

    print(f"\n Searching")
    raw = await call_llm(prompt)
    result = parse_json_response(raw)

    print("\n LLM Reasoning")
    print(result.get("thinking", "")[:600])
    print(f"\nSelected Groups: {result.get('node_list')}")
    return result.get("node_list", [])


async def generate_answer(sections: list[dict], node_ids: list[str], query: str) -> str:
    node_map = {s["id"]: s for s in sections}

    context_parts = []
    for nid in node_ids:
        node = node_map.get(nid)
        if not node:
            continue
        context_parts.append(
            f"=== [{nid}] {node['title']} (page {node['page']}) ===\n{node['text'].strip()}"
        )

    if not context_parts:
        return "NO section found"

    combined = "\n\n".join(context_parts)

    prompt = f"""Answer the question below using ONLY the provided context.
Be concise and accurate. Mention which node IDs you used at the end.

Question: {query}

Context:
{combined}

Answer:"""

    return await call_llm(prompt, temperature=0.1)


# Main
async def main():
    print("=" * 60)
    print("  DIY PageIndex-style Vectorless RAG")
    print("=" * 60)

    pdf_path = download_pdf(PDF_URL)


    print("\nExtracting Document Structure ")
    sections = extract_structure(pdf_path)
    print(f"   {len(sections)} Section Found")
    print("   Sections Sample: ")
    for s in sections[:5]:
        indent = "  " * (s["level"] - 1)
        print(f"   {indent}[{s['id']}] p.{s['page']} — {s['title'][:60]}")


    sections = await build_summaries(sections)


    selected_ids = await search_tree(sections, QUERY)


    print("\n Generate Final Response...")
    answer = await generate_answer(sections, selected_ids, QUERY)

    print("\n" + "═" * 60)
    print("Final Response")
    print("═" * 60)
    print(answer)
    print("═" * 60)


    output = {
        "pdf": PDF_URL,
        "query": QUERY,
        "selected_nodes": selected_ids,
        "answer": answer,
        "tree_summary": [
            {"id": s["id"], "title": s["title"], "page": s["page"], "summary": s["summary"]}
            for s in sections
        ],
    }
    with open("rag_result.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    print("\nResults stored in rag_result.json")


if __name__ == "__main__":
    asyncio.run(main())