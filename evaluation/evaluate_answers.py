"""Lightweight answer evaluation against a human-reviewed answer key.
This computes exact/substring indicators only; semantic faithfulness still needs human review.
"""
import json, re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'reports/assignment_answers.json').read_text(encoding='utf-8'))
key=json.loads((ROOT/'evaluation/answer_key.json').read_text(encoding='utf-8'))
results=[]
for rec in records:
    expected=key[str(rec['id'])]
    answer=rec.get('answer_hi') or ''
    normalized=re.sub(r'\s+','',answer)
    matched=[term for term in expected['must_contain_any'] if re.sub(r'\s+','',term) in normalized]
    results.append({'id':rec['id'],'question':rec['question'],'matched_reference_terms':matched,
                    'reference_term_match':bool(matched),'citation_pages':[c['page'] for c in rec.get('citations',[])],
                    'expected_pages':expected['gold_pages']})
out={'n':len(results),'reference_term_match_rate':sum(x['reference_term_match'] for x in results)/max(1,len(results)),
     'warning':'This is a screening heuristic, not a semantic correctness or faithfulness score. Human review is required.' ,'cases':results}
(ROOT/'reports/answer_evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
