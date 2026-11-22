
"""Run the six assignment queries sequentially with reduced peak RAM usage."""
import gc
import json
import time
from pathlib import Path

import torch

from rag.retriever import HindiRetriever
from rag.generator import HindiGroundedGenerator, select_citations

ROOT = Path(__file__).resolve().parents[1]

questions = json.loads(
    (ROOT / "evaluation/assignment_queries.json").read_text(encoding="utf-8")
)

torch.set_num_threads(1)

# Phase 1: retrieve evidence for all questions using BGE-M3.
print("PHASE 1: Loading BGE-M3 for retrieval...", flush=True)

retriever = HindiRetriever()
retrieved_records = []

for q in questions:
    print(f"Retrieving: {q['id']} - {q['question']}", flush=True)

    start = time.perf_counter()
    hits = retriever.search(q["question"], top_k=5, hybrid=True)
    retrieval_ms = (time.perf_counter() - start) * 1000

    retrieved_records.append({
        "id": q["id"],
        "question": q["question"],
        "retrieval_ms": round(retrieval_ms, 2),
        "hits": hits,
    })

# Release the embedding model before loading Qwen.
print("Releasing BGE-M3 from memory...", flush=True)

del retriever
gc.collect()

if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Phase 2: load Qwen only after retrieval is complete.
print("PHASE 2: Loading Qwen generation model...", flush=True)

generator = HindiGroundedGenerator()
records = []

for item in retrieved_records:
    print(f"Generating answer: {item['id']}", flush=True)

    start = time.perf_counter()
    answer = generator.answer(item["question"], item["hits"])
    generation_ms = (time.perf_counter() - start) * 1000

    citations = select_citations(
        item["question"], answer, item["hits"]
    )

    record = {
        "id": item["id"],
        "question": item["question"],
        "answer_hi": answer,
        "retrieval_ms": item["retrieval_ms"],
        "generation_ms": round(generation_ms, 2),
        "citations": citations,
        "retrieved": [
            {
                "page": h["page"],
                "section": h.get("section", "दस्तावेज़"),
                "chunk_id": h["chunk_id"],
                "score": round(float(h["score"]), 5),
                "source_type": h.get("source_type", "paragraph"),
                "text": h["text"],
            }
            for h in item["hits"]
        ],
    }

    records.append(record)

    print(
        f"\n[{item['id']}] {item['question']}\n"
        f"{answer}\nCitations: {citations}\n",
        flush=True,
    )

# Release generator memory.
del generator
gc.collect()

# Save outputs.
(ROOT / "reports").mkdir(exist_ok=True)

(ROOT / "reports/assignment_answers.json").write_text(
    json.dumps(records, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

lines = [
    "# MegaMindz — six mandatory assignment queries",
    "",
    "Generated using the local ChromaDB index and configured Qwen model.",
    "",
]

for r in records:
    lines.extend([
        f"## {r['id']}. {r['question']}",
        "",
        f"**उत्तर:** {r['answer_hi']}",
        "",
        "**स्रोत:**",
    ])

    for c in r["citations"]:
        lines.append(
            f"- PDF पृष्ठ {c['page']} · अनुभाग: {c['section']} "
            f"· chunk_id: `{c['chunk_id']}`"
        )

    lines.append("")

(ROOT / "reports/assignment_answers.md").write_text(
    "\n".join(lines), encoding="utf-8"
)

print(f"Completed {len(records)} questions. Results saved in reports/.", flush=True)
