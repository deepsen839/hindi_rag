"""Multilingual ChromaDB retrieval with optional Devanagari BM25 fusion."""
from __future__ import annotations
import json, re
from pathlib import Path
import numpy as np
import chromadb
import torch
from rank_bm25 import BM25Okapi
from FlagEmbedding import BGEM3FlagModel
from config import (CHROMA_PATH, BM25_PATH, CHUNKS_PATH, COLLECTION_NAME,
                    EMBEDDING_MODEL, TOP_K, USE_FP16, TORCH_NUM_THREADS, RAG_DEVICE)

DEVANAGARI = re.compile(r'[\u0900-\u097F]')

def tokenize(text: str) -> list[str]:
    return re.findall(r'[\u0900-\u097F]+|[a-zA-Z0-9]+', text.casefold())

class HindiRetriever:
    def __init__(self, use_reranker: bool = False):
        if not CHROMA_PATH.exists() or not BM25_PATH.exists() or not CHUNKS_PATH.exists():
            raise FileNotFoundError('Index missing. Run: python -m rag.ingest && python -m rag.index')
        torch.set_num_threads(TORCH_NUM_THREADS)
        device = RAG_DEVICE if RAG_DEVICE.startswith('cuda') and torch.cuda.is_available() else 'cpu'
        self.device = device
        self.model = BGEM3FlagModel(EMBEDDING_MODEL, use_fp16=bool(USE_FP16 and device != 'cpu'), devices=device)
        self.client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        self.collection = self.client.get_collection(COLLECTION_NAME)
        self.rows = [json.loads(x) for x in CHUNKS_PATH.read_text(encoding='utf-8').splitlines() if x.strip()]
        token_rows = json.loads(BM25_PATH.read_text(encoding='utf-8'))
        self.bm25 = BM25Okapi(token_rows)
        self.reranker = None
        if use_reranker:
            from FlagEmbedding import FlagReranker
            self.reranker = FlagReranker('BAAI/bge-reranker-v2-m3', use_fp16=bool(USE_FP16 and device != 'cpu'), devices=device)

    def search(self, query: str, top_k: int = TOP_K, hybrid: bool = True) -> list[dict]:
        query = (query or '').strip()
        if not query:
            return []
        enc = self.model.encode([query], batch_size=1, max_length=512,
                                return_dense=True, return_sparse=False, return_colbert_vecs=False)
        qvec = np.asarray(enc['dense_vecs'], dtype='float32')[0].tolist()
        n = self.collection.count()
        if n == 0:
            return []
        candidate_k = min(max(top_k * 5, top_k), n)
        raw = self.collection.query(query_embeddings=[qvec], n_results=candidate_k,
                                    include=['documents', 'metadatas', 'distances'])
        dense = {}
        for text, meta, distance in zip(raw['documents'][0], raw['metadatas'][0], raw['distances'][0]):
            item = {'text': text, **meta, 'dense_score': float(1.0 - distance)}
            if item.get('source_type') == 'practice_question':
                continue
            dense[item['chunk_id']] = item
        # BM25 supplements dense retrieval only for Hindi queries; it cannot translate English tokens.
        if hybrid and DEVANAGARI.search(query):
            bm_scores = np.asarray(self.bm25.get_scores(tokenize(query)), dtype='float32')
            max_score = float(bm_scores.max()) if len(bm_scores) else 0.0
            if max_score > 0:
                bm_order = np.argsort(bm_scores)[::-1][:candidate_k]
                for idx in bm_order:
                    row = self.rows[int(idx)]
                    if row.get('source_type') == 'practice_question':
                        continue
                    bm = float(bm_scores[idx] / max_score)
                    cid = row['chunk_id']
                    if cid not in dense:
                        dense[cid] = {**row, 'dense_score': 0.0}
                    dense[cid]['bm25_score'] = bm
        items = list(dense.values())
        for item in items:
            bm = item.get('bm25_score', 0.0)
            item['score'] = 0.80 * item['dense_score'] + 0.20 * bm if hybrid and DEVANAGARI.search(query) else item['dense_score']
        items.sort(key=lambda x: x['score'], reverse=True)
        if self.reranker and items:
            pairs = [[query, x['text']] for x in items[:candidate_k]]
            scores = np.atleast_1d(self.reranker.compute_score(pairs, normalize=True))
            for item, score in zip(items[:candidate_k], scores):
                item['rerank_score'] = float(score)
            items.sort(key=lambda x: x.get('rerank_score', -1), reverse=True)
        return items[:top_k]
