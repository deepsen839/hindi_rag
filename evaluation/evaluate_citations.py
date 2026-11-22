"""Check that every displayed citation is a real retrieved chunk and on a gold page."""
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
records=json.loads((ROOT/'reports/assignment_answers.json').read_text(encoding='utf-8'))
key=json.loads((ROOT/'evaluation/answer_key.json').read_text(encoding='utf-8'))
all_ok=[]; rows=[]
for rec in records:
    expected=set(key[str(rec['id'])]['gold_pages'])
    retrieved={x['chunk_id']:x for x in rec.get('retrieved',[])}
    citations=rec.get('citations',[])
    valid=all(c['chunk_id'] in retrieved and c['page']==retrieved[c['chunk_id']]['page'] and
              c['section']==retrieved[c['chunk_id']]['section'] for c in citations)
    page_hit=any(c['page'] in expected for c in citations)
    ok=bool(citations) and valid and page_hit
    all_ok.append(ok)
    rows.append({'id':rec['id'],'citation_metadata_valid':valid,'has_gold_page_citation':page_hit,
                 'citations':citations})
out={'n':len(rows),'citation_pass_rate':sum(all_ok)/max(1,len(all_ok)),
     'warning':'Page relevance is checked against reviewed gold pages; semantic support requires manual inspection.','cases':rows}
(ROOT/'reports/citation_evaluation.json').write_text(json.dumps(out,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(out,ensure_ascii=False,indent=2))
