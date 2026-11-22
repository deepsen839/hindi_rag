"""FastAPI interface for the cross-lingual Hindi RAG pipeline."""
from threading import Lock
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from rag.retriever import HindiRetriever
from rag.generator import HindiGroundedGenerator, select_citations

app = FastAPI(title='MegaMindz AI Solutions — Hindi RAG', version='1.0.0')
retriever = generator = None
lock = Lock()

class QueryRequest(BaseModel):
    question: str = Field(min_length=2, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=10)
    generate: bool = True
    hybrid: bool = True

def get_retriever():
    global retriever
    if retriever is None:
        with lock:
            if retriever is None:
                retriever = HindiRetriever()
    return retriever

def get_generator():
    global generator
    if generator is None:
        with lock:
            if generator is None:
                generator = HindiGroundedGenerator()
    return generator

@app.get('/health')
def health():
    return {'status':'ok','vector_database':'ChromaDB','retriever_loaded':retriever is not None,
            'generator_loaded':generator is not None}

@app.post('/query')
def query(req: QueryRequest):
    try:
        hits = get_retriever().search(req.question, top_k=req.top_k, hybrid=req.hybrid)
        if not hits:
            return {'question':req.question,'answer_hi':'दिए गए दस्तावेज़ में संबंधित साक्ष्य नहीं मिला。',
                    'citations':[],'retrieved':[]}
        answer = get_generator().answer(req.question, hits) if req.generate else None
        citations = select_citations(req.question, answer or '', hits) if req.generate else []
        return {'question':req.question,'answer_hi':answer,'citations':citations,
                'retrieved':[{'page':h['page'],'section':h.get('section','दस्तावेज़'),
                              'chunk_id':h['chunk_id'],'score':round(float(h['score']),5),
                              'source_type':h.get('source_type','paragraph'),'text':h['text']} for h in hits]}
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f'RAG request failed: {type(exc).__name__}: {exc}') from exc
