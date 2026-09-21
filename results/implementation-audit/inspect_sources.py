"""Record unchanged graph kernels and narrowly normalized query-body comparisons."""
from pathlib import Path
import hashlib,json,re,subprocess
OUT=Path(__file__).resolve().parent;HVS=OUT.parents[1]
UP=Path('/home/weitang/hvs-audit-20260921/upstream')
def clean(s):
 return re.sub(r'//[^\n]*|/\*[\s\S]*?\*/','',s)
def compact(s):return re.sub(r'\s+','',clean(s))
def bodies(s,name):
 s=clean(s);result=[]
 for m in re.finditer(r'\b'+re.escape(name)+r'\s*\(',s):
  start=m.end()-1;depth=0;end=start
  for i in range(start,len(s)):
   if s[i]=='(':depth+=1
   elif s[i]==')':depth-=1
   if depth==0:end=i+1;break
  tail=re.match(r'\s*(?:const\s*)?(?:noexcept\s*)?\{',s[end:])
  if not tail:continue
  begin=end+tail.end()-1;depth=0
  for i in range(begin,len(s)):
   if s[i]=='{':depth+=1
   elif s[i]=='}':depth-=1
   if depth==0:
    result.append(compact(s[begin:i+1]));break
 return result
u=(UP/'hnsw/hnswlib/hnswalg.h').read_text();c=(HVS/'hnsw/hnswlib/hnswalg.h').read_text()
names=['getRandomLevel','InsertIntoPool','searchBaseLayer','getNeighborsByHeuristic2','mutuallyConnectNewElement','connect','searchQuanST','est_density','permutation','quandistfunc_','querydistfunc_','addPoint','getDataByInternalId','getExternalLabel']
report={'unchanged_graph_function_bodies':{},'query_function_bodies':{},'unchanged_files':{},'previously_validated_sources':{}}
for name in names:
 ub,cb=bodies(u,name),bodies(c,name)
 report['unchanged_graph_function_bodies'][name]={'upstream_definitions':len(ub),'current_definitions':len(cb),'equal_ignoring_comments_whitespace':bool(ub) and ub==cb}
alloc=compact('std::vector<Neighbor> local_candidates; auto& retset = workspace ? *workspace : local_candidates; retset.assign(LL + 1, Neighbor{});')
for name in ['SearchWithsingleGraph','SearchWithquanGraph','SearchWithquanGraph2','SearchWithquanGraph3','SearchWithOptGraph']:
 ub=bodies(u,name);cb=bodies(c,name)
 assert len(ub)==len(cb)==1
 norm_old=ub[0].replace(compact('std::vector<unsigned> init_ids(LL);'),'')
 norm_new=cb[0].replace(alloc,compact('std::vector<Neighbor> retset(LL + 1);'))
 report['query_function_bodies'][name]={'equal_after_candidate_storage_normalization':norm_old==norm_new,'normalization':'Replace reusable workspace selection/reset with original local candidate allocation; remove original unused init_ids vector. No arithmetic, comparisons, graph traversal or result mapping normalized.'}
for p in (UP/'hnsw/hnswlib').glob('*.h'):
 if p.name!='hnswalg.h':report['unchanged_files'][str(p.relative_to(UP))]=p.read_bytes()==(HVS/p.relative_to(UP)).read_bytes()
old=json.loads(Path('/home/weitang/hvs-equivalence-20260921/source-manifest.json').read_text())['current_sha256']
for path,expected in old.items():
 if path.startswith('hnsw/') and Path(path).suffix in ['.cpp','.hpp','.h']:
  report['previously_validated_sources'][path]=hashlib.sha256((HVS/path).read_bytes()).hexdigest()==expected
(OUT/'source-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
assert all(r['equal_ignoring_comments_whitespace'] for r in report['unchanged_graph_function_bodies'].values())
assert all(r['equal_after_candidate_storage_normalization'] for r in report['query_function_bodies'].values())
assert all(report['unchanged_files'].values()) and all(report['previously_validated_sources'].values())
