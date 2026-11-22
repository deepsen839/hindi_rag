"""End-to-end API benchmark. Measures warm request latency; run cold start separately."""
import argparse, json, statistics, time, requests
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--url',default='http://localhost:8000/query'); p.add_argument('--runs',type=int,default=10); p.add_argument('--generate',action='store_true'); args=p.parse_args()
qs=[x['question'] for x in json.loads((Path(__file__).parent/'assignment_queries.json').read_text(encoding='utf-8'))]
for q in qs: requests.post(args.url,json={'question':q,'generate':args.generate},timeout=180).raise_for_status()
lat=[]
for _ in range(args.runs):
 for q in qs:
  t=time.perf_counter(); r=requests.post(args.url,json={'question':q,'generate':args.generate},timeout=180); r.raise_for_status(); lat.append((time.perf_counter()-t)*1000)
lat.sort(); result={'requests':len(lat),'generation_enabled':args.generate,'mean_ms':statistics.mean(lat),'p50_ms':statistics.median(lat),'p95_ms':lat[min(len(lat)-1,int(.95*(len(lat)-1)))],'min_ms':lat[0],'max_ms':lat[-1]}
Path('reports').mkdir(exist_ok=True); Path('reports/latency_benchmark.json').write_text(json.dumps(result,indent=2),encoding='utf-8'); print(json.dumps(result,indent=2))
