# AI Solutions — Cross-Lingual Hindi RAG

A Docker-based Retrieval-Augmented Generation (RAG) project built over the supplied **Agni Ki Udaan Hindi study guide**. The system accepts Hindi and English questions, retrieves relevant Hindi evidence, generates grounded Hindi answers, and returns source citations.

**Core components**

- **PDF processing:** PyMuPDF
- **Vector database:** persistent ChromaDB
- **Multilingual embeddings:** `BAAI/bge-m3`
- **Lexical retrieval:** BM25, combined with dense retrieval for hybrid search
- **Answer generation:** `Qwen/Qwen2.5-1.5B-Instruct`
- **API:** FastAPI
- **Optional interface:** Streamlit
- **Containerization:** Docker Engine and Docker Compose

> **Important execution note:** The supplied Dockerfile installs CPU-only PyTorch. The commands in this README explain how to run the project; they do not claim that full model inference, benchmark results, or performance targets have been verified on every machine. First-time model downloads and CPU inference can take a long time and require substantial RAM and disk space.

---

## 1. Features

- Page-wise PDF text extraction.
- Unicode NFC normalization and Devanagari-aware text handling.
- Hindi danda (`।`) and paragraph-aware text processing.
- Heading-/paragraph-aware chunking with page, section, and chunk ID metadata.
- Table-row reconstruction to keep relevant values and labels together where detected.
- Multilingual dense retrieval: English questions can retrieve Hindi passages.
- Hybrid retrieval combining dense vectors and BM25 lexical search.
- Persistent ChromaDB vector storage.
- Grounded Hindi answer generation using retrieved document evidence.
- Citation metadata containing PDF page, section, and chunk ID.
- FastAPI `/query` endpoint and optional Streamlit UI.
- Retrieval, answer, citation, and API latency evaluation scripts.

## 2. Architecture

```text
                    Supplied Hindi PDF
                            |
                            v
                 PyMuPDF page-wise extraction
                            |
                            v
             Unicode NFC + Hindi text normalization
                            |
                            v
              Paragraph / heading / table processing
                            |
                            v
              Chunking + source metadata assignment
                            |
                            v
                   BAAI/bge-m3 embeddings
                            |
                            v
             Persistent ChromaDB vector collection
                            |
                 +----------+-----------+
                 |                      |
                 v                      v
           Dense retrieval          BM25 retrieval
                 |                      |
                 +----------+-----------+
                            |
                            v
                   Hybrid retrieval
                            |
                            v
                 Relevant Hindi chunks
                            |
                            v
                Qwen2.5-1.5B-Instruct
                            |
                            v
              Grounded answer + citations
                 (page/section/chunk ID)
```

### Hindi and document processing

- PDF pages are processed separately.
- Page numbers in metadata are 1-based physical PDF page numbers.
- Text is normalized to Unicode NFC; Devanagari text is not transliterated.
- Paragraphs and headings are used to retain section context.
- Table rows are represented as evidence units where detected by the extraction pipeline.
- Chunks carry source metadata such as page, section, chunk ID, language, and source type.

### Retrieval and generation

`BAAI/bge-m3` is the multilingual embedding model used to represent Hindi and English text in a shared embedding space. ChromaDB stores the dense vectors. BM25 adds lexical matching for hybrid retrieval. Qwen2.5-1.5B-Instruct generates an answer using retrieved passages. The citation metadata identifies the source chunks; citation metadata checks alone do not prove that every generated claim is semantically supported.

---

## 3. Project structure

```text
hindi_complete/
├── api.py
├── app.py
├── config.py
├── Dockerfile
├── compose.yaml
├── requirements.txt
├── requirements-local.txt
├── .env.docker.example
├── .env.example
├── rag/
│   ├── ingest.py
│   ├── index.py
│   ├── retriever.py
│   └── generator.py
├── evaluation/
│   ├── assignment_queries.json
│   ├── answer_key.json
│   ├── gold_qa.json
│   ├── expected_answers.md
│   ├── run_assignment_queries.py
│   ├── evaluate_retrieval.py
│   ├── evaluate_answers.py
│   ├── evaluate_citations.py
│   └── benchmark.py
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
└── reports/
```

The `artifacts/`, `data/processed/`, and `reports/` directories hold runtime data. Their contents are created or updated when the pipeline and evaluation commands run.

---

## 4. Requirements

- Ubuntu/Linux or another Docker-compatible operating system.
- Docker Engine and the Docker Compose plugin.
- Internet access for the first download of Hugging Face model weights.
- Enough free disk space for Docker images, model caches, the vector index, and reports.
- Enough RAM for the embedding model and (when answer generation is enabled) the Qwen generation model.

### CPU and GPU note

The included Dockerfile installs **CPU-only PyTorch** and the Compose configuration sets `RAG_DEVICE=cpu`. This default setup does not use an NVIDIA GPU. GPU execution requires a separate CUDA-compatible Docker/PyTorch configuration and a working NVIDIA Container Toolkit installation.

### Memory guidance

Model loading and inference can consume several gigabytes of RAM. On a low-memory machine:

1. Do not run API, Streamlit, ingestion, assignment generation, and generation benchmarks simultaneously.
2. Stop the API/UI before running the one-off assignment script if they are not needed.
3. Use the one-thread environment overrides shown below to reduce CPU thread contention.
4. Monitor memory with `free -h`, `swapon --show`, and `docker stats`.

Thread limits reduce CPU contention; **they do not cap RAM usage**. Swap can help avoid an out-of-memory kill but is much slower than physical RAM.

---

## 5. Install Docker on Ubuntu

If Docker is already installed and working, skip this section. Install Docker Engine and the Compose plugin using Docker's official Ubuntu repository instructions. After installation, verify:

```bash
sudo systemctl enable --now docker
docker --version
docker compose version
sudo docker run hello-world
```

If Docker only works with `sudo`, add your user to the Docker group:

```bash
sudo usermod -aG docker "$USER"
```

Log out and log back in (or start a new login session) before using Docker without `sudo`. Membership in the Docker group grants root-equivalent privileges on the host; only add trusted users.

---

## 6. Run the project with Docker Compose

Run the commands from the project root, the directory containing `compose.yaml`.


### Step 1 — Create the environment file

```bash
cp .env.docker.example .env
```

Review `.env`. The default values include:

```env
RAG_PDF_PATH=/app/data/raw/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf
EMBEDDING_MODEL=BAAI/bge-m3
GENERATION_MODEL=Qwen/Qwen2.5-1.5B-Instruct
CHROMA_COLLECTION=agni_ki_udaan_hindi
RAG_DEVICE=cpu
USE_FP16=false
TORCH_NUM_THREADS=4
EMBEDDING_BATCH_SIZE=1
TOP_K=5
CHUNK_WORDS=220
CHUNK_OVERLAP=45
MAX_CONTEXT_CHARS=3500
MAX_NEW_TOKENS=128
```

Ensure the PDF exists at the configured path inside the project. The provided Compose setup mounts the project into the tools container and mounts the configured persistent data locations into the API/UI containers.

### Step 2 — Build the Docker image

```bash
docker compose build
```

**What happens during `docker compose build`:** Docker builds the image, installs OS/Python dependencies, and copies the project source. It does **not** automatically run PDF ingestion, create the ChromaDB index, run the six assignment questions, or execute evaluation scripts.

### Step 3 — Ingest the PDF, build the index, and evaluate retrieval

```bash
docker compose --profile tools run --rm pipeline
```

The `pipeline` service runs these commands sequentially:

```bash
python -m rag.ingest
python -m rag.index
python -m evaluation.evaluate_retrieval
```

The `&&` in the Compose command means the next step runs only if the previous step succeeds. Run this pipeline again if the source PDF or relevant ingestion/index settings change.

### Step 4 — Start the API

```bash
docker compose up -d api
docker compose ps
```

Follow the API logs:

```bash
docker compose logs -f api
```

Health endpoint:

```bash
curl http://localhost:8000/health
```

Interactive API documentation:

```text
http://localhost:8000/docs
```

The API health response reports whether the retriever and generator have been loaded. The models are loaded lazily when the first applicable query arrives.

### Step 5 — Send a Hindi query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"कलाम को भारत रत्न किस वर्ष प्राप्त हुआ?","top_k":5,"generate":true,"hybrid":true}'
```

### Step 6 — Send an English cross-lingual query

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Which institution did Kalam attend to study aeronautical engineering?","top_k":5,"generate":true,"hybrid":true}'
```

The `/query` request accepts:

- `question`: Hindi or English question text.
- `top_k`: number of retrieved passages (1–10; default 5).
- `generate`: whether to load/use Qwen and generate an answer (default `true`).
- `hybrid`: whether to use hybrid retrieval (default `true`).

Use `"generate": false` to test retrieval without loading the Qwen generation model. The response includes retrieved passages and their source metadata; generated answers include citation metadata.

---

## 7. Run the six mandatory assignment questions

The six questions are defined in `evaluation/assignment_queries.json`.

### Step 1 — Avoid running multiple model services at once

If your machine has limited RAM, stop API and UI before running the assignment container:

```bash
docker compose stop api ui
```

### Step 2 — Run the assignment service

```bash
docker compose --profile tools run --rm \
  -e TORCH_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  assignment
```

This invokes the service's default command:

```bash
python -m evaluation.run_assignment_queries
```

The three `-e` options set environment variables for this one-off container:

- `TORCH_NUM_THREADS=1`: limits PyTorch intra-op CPU threads.
- `OMP_NUM_THREADS=1`: limits OpenMP threads.
- `MKL_NUM_THREADS=1`: limits MKL numerical-computation threads.

`--profile tools` enables the tools-profile service, `run` creates a one-off container, and `--rm` removes that container after it exits. The bind-mounted project files and persistent data are not removed by `--rm`.

### Step 3 — Review generated answers and citations

On successful completion, the runner writes:

```text
reports/assignment_answers.json
reports/assignment_answers.md
```

These are actual runtime outputs. `evaluation/expected_answers.md` and `evaluation/answer_key.json` are evaluation/reference material; they are not generated model answers.

> If the machine freezes or the process is killed, do not repeatedly rerun it. Check memory and Docker logs first. Depending on the exact runner version in your checkout, retrieval and generation models may remain loaded together; if peak RAM is too high, run the assignment on a machine with more RAM or update the runner to retrieve all questions first, release the embedding model, and then load the generation model.

---

## 8. Run the evaluation scripts inside Docker

The evaluation commands below run inside the `assignment` service container. They override that service's default command for that invocation. Run them **one at a time**, from the project root.

### A. Retrieval evaluation

```bash
docker compose --profile tools run --rm \
  -e TORCH_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  assignment python -m evaluation.evaluate_retrieval
```

This loads the retriever and evaluates the reviewed page-level gold set in `evaluation/gold_qa.json`. It writes:

```text
reports/retrieval_results.json
```

Note: the `pipeline` service already runs this retrieval evaluation after ingestion and indexing. You can run it separately whenever you want to repeat the evaluation against the existing index.

### B. Run the six assignment questions

```bash
docker compose --profile tools run --rm \
  -e TORCH_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  assignment python -m evaluation.run_assignment_queries
```

Writes:

```text
reports/assignment_answers.json
reports/assignment_answers.md
```

### C. Evaluate generated answers

Run this **after** the assignment query runner has created `reports/assignment_answers.json`:

```bash
docker compose --profile tools run --rm assignment \
  python -m evaluation.evaluate_answers
```

Writes:

```text
reports/answer_evaluation.json
```

This is a reference-term/substring screening heuristic. It is not a semantic correctness score or a proof of faithfulness.

### D. Evaluate citations

Run this **after** the assignment query runner has created `reports/assignment_answers.json`:

```bash
docker compose --profile tools run --rm assignment \
  python -m evaluation.evaluate_citations
```

Writes:

```text
reports/citation_evaluation.json
```

The script checks that cited chunk IDs and metadata correspond to retrieved records and that citations include a reviewed gold page. This is not a substitute for manually verifying that each answer claim is supported by its cited passage.

### E. Run the API latency benchmark (10 rounds)

**Important:** Unlike the preceding evaluation scripts, `evaluation.benchmark` sends HTTP requests to the FastAPI service. The API must be running while the benchmark executes.

Start the API:

```bash
docker compose up -d api
```

Check it from the host:

```bash
curl http://localhost:8000/health
```

Then run the benchmark container using the Compose service hostname `api`:

```bash
docker compose --profile tools run --rm \
  -e TORCH_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  assignment \
  python -m evaluation.benchmark \
  --url http://api:8000/query \
  --runs 10
```

**Why `http://api:8000/query` and not `http://localhost:8000/query`?** The benchmark runs inside its own container. Inside that container, `localhost` refers to the benchmark container itself, not the API container. Docker Compose provides service-name DNS, so `api` resolves to the API service on the Compose network.

The benchmark script performs one warm-up request for each of the six questions, then sends six questions per measured round. With `--runs 10`, that is 6 warm-up requests plus 60 measured requests (66 total). It reports mean, median/p50, p95, minimum, and maximum request latency. It writes:

```text
reports/latency_benchmark.json
```

By default, the benchmark sends `generate=false` because the script only sets `generate` to true when `--generate` is supplied. This measures the API retrieval path without Qwen answer generation.

### F. Optional benchmark with answer generation enabled

```bash
docker compose --profile tools run --rm \
  -e TORCH_NUM_THREADS=1 \
  -e OMP_NUM_THREADS=1 \
  -e MKL_NUM_THREADS=1 \
  assignment \
  python -m evaluation.benchmark \
  --url http://api:8000/query \
  --runs 10 \
  --generate
```

This causes the API to load/use Qwen and generate an answer for each measured request. It can be substantially slower and use more memory. Do not run this at the same time as the standalone assignment generation process on a low-memory machine.

### Recommended evaluation order

```text
1. docker compose build
2. docker compose --profile tools run --rm pipeline
3. Start API: docker compose up -d api
4. Retrieval evaluation (optional repeat; pipeline already ran it)
5. Assignment queries (run separately; stop API first if RAM is limited)
6. Answer evaluation
7. Citation evaluation
8. API benchmark (API must be running; use --url http://api:8000/query)
```

For memory-constrained systems, do not keep the API running while running the standalone assignment query generator. Stop/start the API as needed. The API benchmark is the exception: it requires the API service to be running.

---

## 9. Run the Streamlit UI

The Streamlit UI is an alternative interaction mode. On a memory-constrained machine, do not run it alongside the API or the assignment generation process unless you have enough RAM.

```bash
docker compose stop api
docker compose --profile ui up ui
```

Open:

```text
http://localhost:8501
```

Stop the foreground process with `Ctrl+C`. If it was started detached, stop it with:

```bash
docker compose --profile ui down
```

---

## 10. Run locally without Docker (optional)

Python 3.11 is recommended. Create and activate a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements-local.txt
cp .env.example .env
```

Ensure `RAG_PDF_PATH` in `.env` points to the PDF file on the host. Then run ingestion, indexing, and retrieval evaluation:

```bash
python -m rag.ingest
python -m rag.index
python -m evaluation.evaluate_retrieval
```

Start the API:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Or run the UI in a separate terminal:

```bash
streamlit run app.py --server.port 8501
```

---

## 11. Output files and persistence

| Path | Purpose |
|---|---|
| `data/raw/` | Source PDF input |
| `data/processed/` | Extracted text/chunks and ingestion metadata |
| `artifacts/` | Persistent ChromaDB database files |
| `reports/retrieval_results.json` | Page-level retrieval metrics and cases |
| `reports/assignment_answers.json` | Generated answers, retrieved chunks, and citation metadata |
| `reports/assignment_answers.md` | Human-readable assignment answer report |
| `reports/answer_evaluation.json` | Reference-term screening results |
| `reports/citation_evaluation.json` | Citation metadata/page checks |
| `reports/latency_benchmark.json` | API latency benchmark summary |
| Docker volume `hf_cache` | Cached Hugging Face model files |

The tools and assignment services mount the project directory into `/app`, so files written under `/app/reports` are written into the host project's `reports/` directory.

---

## 12. Troubleshooting

### A. Docker daemon is not running

```bash
sudo systemctl status docker
sudo systemctl start docker
sudo systemctl enable docker
sudo docker info
```

### B. Container name is already in use

List running and stopped containers:

```bash
docker ps -a --filter name=hindi-rag-api
```

If the old container is confirmed to be from this project and is no longer needed, stop and remove it:

```bash
docker stop hindi-rag-api
docker rm hindi-rag-api
```

Removing a container does not remove bind-mounted project files. Avoid deleting volumes unless you intend to remove their data.

### C. `Connection refused` from the benchmark

The benchmark needs the API service. Start it:

```bash
docker compose up -d api
curl http://localhost:8000/health
```

From the benchmark container, use the Compose service hostname:

```bash
--url http://api:8000/query
```

Do not use `localhost:8000` inside the benchmark container to reach the separate API container.

### D. Machine freezes, OOM, or container exits

Inspect host memory and swap:

```bash
free -h
swapon --show
docker stats
```

Inspect container state and logs:

```bash
docker compose ps -a
docker compose logs --tail=100 api
```

Look for Linux OOM-killer messages:

```bash
sudo journalctl -k --since "30 minutes ago" | grep -iE 'oom|killed process'
```

Stop services that are not needed. Run one model-heavy job at a time. The Compose `mem_limit` is a container memory ceiling, not a guarantee that the host has enough RAM for all running containers.

### E. Transformers / PEFT FutureWarning

A `FutureWarning` about PEFT and Transformers is a compatibility warning, not necessarily a runtime failure. The project requirements constrain Transformers below major version 5. Do not blindly upgrade Transformers to satisfy a warning: check the installed versions and package constraints first.

Check installed versions inside a running API container:

```bash
docker compose exec api python -c \
  "import transformers; print('Transformers:', transformers.__version__); import peft; print('PEFT:', peft.__version__)"
```

If PEFT is not directly used by your project, confirm whether it is being installed transitively before changing dependencies. Rebuild after deliberate dependency changes:

```bash
docker compose build --no-cache
```

### F. Rebuild after source/dependency changes

```bash
docker compose build
```

Use `--no-cache` only when you specifically need a clean rebuild:

```bash
docker compose build --no-cache
```

Do **not** use `docker compose down -v` as a routine cleanup command. The `-v` option removes named volumes, including the Hugging Face cache volume used by this project.

### G. Inspect container logs and status

```bash
docker compose ps -a
docker compose logs --tail=150 api
```

For a one-off command, Docker prints the command's stdout/stderr in the terminal. If it exits unexpectedly, inspect the traceback and the host's memory state before retrying.

---

## 13. Evaluation interpretation and limitations

- Retrieval evaluation uses a reviewed starter gold set and page-level relevance labels. Its metrics include Recall@5, Precision@5, MRR@5, nDCG@5, and retrieval latency.
- Answer evaluation uses reference terms and substring matching. It is a screening heuristic, not a semantic answer-quality or faithfulness metric.
- Citation evaluation checks citation metadata against retrieved records and reviewed gold pages. It does not establish claim-level semantic support.
- The API benchmark measures warm HTTP request latency after warm-up requests. It does not represent cold model startup latency. Retrieval-only and generation-enabled results are different workloads and should be reported separately.
- Manually inspect retrieved passages and citations before making accuracy or faithfulness claims.
- Scanned/image-only PDFs require OCR; this pipeline is designed for the supplied text-extractable PDF.
- The development API has no authentication or rate limiting. Do not expose it publicly without security controls.

---

## 14. Security and data handling

- Do not commit `.env`, access tokens, credentials, model weights, or local database artifacts to a public repository.
- The Hugging Face models are downloaded at runtime/first use and cached in the Docker `hf_cache` volume.
- Check the licenses and terms for the supplied PDF, third-party models, and dependencies before redistribution or commercial deployment.

## 15. Suggested future improvements

- Expand the manually reviewed Hindi/English evaluation set.
- Add claim-level faithfulness evaluation with human review.
- Add OCR and visual table extraction quality checks for scanned documents.
- Compare dense-only, BM25-only, and hybrid retrieval.
- Test multilingual reranking and tune chunk size/overlap.
- Add API authentication, rate limiting, observability, and production deployment configuration.

---

**Project:** AI Solutions — Cross-Lingual Hindi RAG  
**Primary database:** ChromaDB  
**Embedding model:** BAAI/bge-m3  
**Generation model:** Qwen2.5-1.5B-Instruct  
**Default runtime:** Docker Compose, CPU
