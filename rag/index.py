"""Create a persistent ChromaDB collection using BGE-M3 multilingual vectors."""
import json, re
import chromadb
import torch
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi
from config import (CHUNKS_PATH, CHROMA_PATH, BM25_PATH, ARTIFACT_DIR, COLLECTION_NAME,
                    EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE, USE_FP16, TORCH_NUM_THREADS, RAG_DEVICE)

def load_chunks():
    if not CHUNKS_PATH.exists():
        raise FileNotFoundError(f'Missing {CHUNKS_PATH}; run `python -m rag.ingest` first')
    with CHUNKS_PATH.open(encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]

def tokenize_hindi(text):
    return re.findall(r'[\u0900-\u097F\w]+', text.lower())

def main():
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    chunks = load_chunks()
    if not chunks:
        raise RuntimeError('No chunks found. Run python -m rag.ingest first.')
    torch.set_num_threads(TORCH_NUM_THREADS)
    device = RAG_DEVICE if RAG_DEVICE.startswith('cuda') and torch.cuda.is_available() else 'cpu'
    print(f'Loading {EMBEDDING_MODEL} on {device}; chunks={len(chunks)}', flush=True)
    model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=bool(USE_FP16 and device != 'cpu'), devices=device)
    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    collection = client.create_collection(name=COLLECTION_NAME, metadata={'hnsw:space': 'cosine', 'embedding_model': EMBEDDING_MODEL})
    dimension = None
    for start in range(0, len(chunks), EMBEDDING_BATCH_SIZE):
        batch = chunks[start:start + EMBEDDING_BATCH_SIZE]
        encoded = model.encode([x['text'] for x in batch], batch_size=EMBEDDING_BATCH_SIZE, max_length=512,
                               return_dense=True, return_sparse=False, return_colbert_vecs=False)
        vectors = encoded['dense_vecs'].tolist()
        dimension = len(vectors[0])
        metadatas = [{k: v for k, v in row.items() if k != 'text'} for row in batch]
        collection.add(ids=[x['chunk_id'] for x in batch], embeddings=vectors,
                       documents=[x['text'] for x in batch], metadatas=metadatas)
        print(f'Embedded and stored {min(start + len(batch), len(chunks))}/{len(chunks)}', flush=True)
    token_rows = [tokenize_hindi(x['text']) for x in chunks]
    BM25_PATH.write_text(json.dumps(token_rows, ensure_ascii=False), encoding='utf-8')
    info = {'vector_database': 'ChromaDB', 'collection': COLLECTION_NAME,
            'embedding_model': EMBEDDING_MODEL, 'dimension': int(dimension or 0),
            'vectors': collection.count(), 'distance_metric': 'cosine', 'device': device, 'dtype': 'float32',
            'metadata_fields': ['page','section','chunk_id','source_type','language','script']}
    (ARTIFACT_DIR / 'index_info.json').write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
