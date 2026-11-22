"""Page-grounded retrieval metrics for the reviewed starter gold set."""
import json, math, statistics
from pathlib import Path
from rag.retriever import HindiRetriever
ROOT=Path(__file__).resolve().parents[1]
gold=json.loads((ROOT/'evaluation/gold_qa.json').read_text(encoding='utf-8'))
r=HindiRetriever(); rows=[]; recalls=[]; precisions=[]; mrrs=[]; ndcgs=[]; latencies=[]
import time
for q in gold:
    t=time.perf_counter(); hits=r.search(q['question'],top_k=5,hybrid=True); ms=(time.perf_counter()-t)*1000
    pages=[int(h['page']) for h in hits]; relevant=set(q['gold_pages'])
    rel=[1 if p in relevant else 0 for p in pages]
    hit_positions=[i+1 for i,x in enumerate(rel) if x]
    recalls.append(1.0 if hit_positions else 0.0)
    precisions.append(sum(rel)/5)
    mrrs.append(1/min(hit_positions) if hit_positions else 0.0)
    dcg=sum(x/math.log2(i+2) for i,x in enumerate(rel))
    ideal=sum(1/math.log2(i+2) for i in range(min(len(relevant),5)))
    ndcgs.append(dcg/ideal if ideal else 0.0); latencies.append(ms)
    rows.append({'id':q['id'],'question':q['question'],'gold_pages':sorted(relevant),
                 'retrieved_pages':pages,'retrieved_chunk_ids':[h['chunk_id'] for h in hits],
                 'hit_at_5':bool(hit_positions),'reciprocal_rank':mrrs[-1], 'latency_ms':round(ms,2)})
summary={'n_questions':len(gold),'recall_at_5':statistics.mean(recalls),
         'precision_at_5_page_level':statistics.mean(precisions),'mrr_at_5':statistics.mean(mrrs),
         'ndcg_at_5_page_level':statistics.mean(ndcgs),'retrieval_latency_ms_mean':statistics.mean(latencies),
         'retrieval_latency_ms_p95':sorted(latencies)[min(len(latencies)-1,math.ceil(.95*len(latencies))-1)],
         'note':'Gold relevance is physical PDF page-level; inspect cases and expand reviewed chunk-level labels.'}
(ROOT/'reports').mkdir(exist_ok=True)
(ROOT/'reports/retrieval_results.json').write_text(json.dumps({'summary':summary,'cases':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(summary,ensure_ascii=False,indent=2))
