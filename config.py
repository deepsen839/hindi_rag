from pathlib import Path
import os
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / '.env')
PDF_PATH = Path(os.getenv('RAG_PDF_PATH', str(ROOT / 'data/raw/Agni_Ki_Udaan_Adhyayan_Sahayika_Hindi.pdf')))
if not PDF_PATH.is_absolute():
    PDF_PATH = ROOT / PDF_PATH
DATA_DIR = ROOT / 'data' / 'processed'
ARTIFACT_DIR = ROOT / 'artifacts'
CHUNKS_PATH = DATA_DIR / 'chunks.jsonl'
CHROMA_PATH = ARTIFACT_DIR / 'chroma_db'
BM25_PATH = ARTIFACT_DIR / 'bm25_tokens.json'
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION', 'agni_ki_udaan_hindi')
EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
RERANKER_MODEL = os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-v2-m3')
GENERATION_MODEL = os.getenv('GENERATION_MODEL', 'Qwen/Qwen2.5-1.5B-Instruct')
CHUNK_WORDS = int(os.getenv('CHUNK_WORDS', '220'))
CHUNK_OVERLAP = int(os.getenv('CHUNK_OVERLAP', '45'))
TOP_K = int(os.getenv('TOP_K', '5'))
MAX_CONTEXT_CHARS = int(os.getenv('MAX_CONTEXT_CHARS', '3500'))
RAG_DEVICE = os.getenv('RAG_DEVICE', 'cpu').lower()
EMBEDDING_BATCH_SIZE = int(os.getenv('EMBEDDING_BATCH_SIZE', '1'))
TORCH_NUM_THREADS = int(os.getenv('TORCH_NUM_THREADS', '4'))
USE_FP16 = os.getenv('USE_FP16', 'false').lower() in {'1', 'true', 'yes'}
MAX_NEW_TOKENS = int(os.getenv('MAX_NEW_TOKENS', '128'))
