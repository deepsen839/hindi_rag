# MegaMindz Hindi RAG — implementation and validation report

## Source document

`Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf` was inspected directly. It contains 22 physical PDF pages. The current ingestion run detected 132 paragraph units, 30 table rows and generated 162 chunks. The generated `data/processed/ingestion_stats.json` and `chunks.jsonl` are included for inspection.

## Implemented components

- Page-wise PyMuPDF text extraction and table-row reconstruction for awards, timeline and quick-fact tables.
- Unicode NFC normalization and UTF-8 storage.
- Numbered-section/paragraph-aware chunking with 220-word windows and 45-word overlap.
- Practice-question chunk labelling to reduce question-only retrieval false positives.
- BAAI/bge-m3 multilingual dense embeddings.
- ChromaDB persistent vector storage with cosine distance and page/section/chunk metadata.
- Devanagari BM25 fusion for Hindi queries; multilingual dense retrieval for English queries.
- Qwen2.5-1.5B-Instruct evidence-grounded Hindi generation.
- Citation metadata selected programmatically from retrieved chunks.
- FastAPI, Streamlit, Docker Compose and retrieval/answer/citation/latency evaluation scripts.

## Source-verified mandatory answers

See `evaluation/expected_answers.md`. It records the correct facts and page/chunk references as a human-reviewed answer key. It is not presented as an LLM runtime log. Run `python -m evaluation.run_assignment_queries` to generate actual model outputs and citations.

## Validation performed in this packaging environment

- PDF extraction executed successfully.
- 22 pages, 132 paragraph units, 30 table rows and 162 chunks generated.
- Four offline unit tests passed.
- Python source files passed `py_compile`.

## Not yet executed here

The container used to package this repository does not have ChromaDB, FlagEmbedding, Transformers or Streamlit installed and cannot establish a full model inference environment. Therefore the Chroma vector index build, BGE-M3 embedding run, Qwen answer generation, API/UI runtime and retrieval/latency scores have **not** been claimed as executed. Install `requirements.txt`, build the index and run the evaluation commands in README on your machine. Runtime outputs are written to `reports/`.

## Known limitations

- Table extraction is a PDF-specific text reconstruction fallback; inspect table rows against the source pages.
- Gold retrieval labels are page-level starter labels; chunk-level labels should be reviewed for a stronger evaluation.
- Citation metadata validity is tested programmatically, but semantic claim-to-evidence entailment still needs human review.
- CPU generation may be slow; GPU configuration is optional and hardware-dependent.
