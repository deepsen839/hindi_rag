# Faithfulness evaluation protocol

Do not label retrieval recall as faithfulness. Evaluate generated answers separately.

For each gold question:
1. Retrieve top-5 Hindi passages.
2. Generate a Hindi answer with page citations.
3. Split the answer into atomic factual claims.
4. For every claim, verify that at least one cited/retrieved passage directly supports it.
5. Mark unsupported claims, incorrect citations, and omitted uncertainty.
6. Score = supported factual claims / total factual claims. If no factual claims are produced, mark the case invalid rather than scoring 1.0.

Report both:
- Human-reviewed faithfulness (primary; target >= 0.85).
- Optional LLM-judge estimate (secondary; never substitute for human review).

Also report citation precision = citations pointing to supporting pages / all citations.
Keep raw questions, retrieved chunks, generated answers, citations, judge rationale and human labels in JSONL.
