"""Compare fresh maximum-thread builds and both query engines on the same index."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,time
import numpy as np
OUT=Path(__file__).resolve().parent
REPO=OUT.parents[3]
WORK=Path('/home/weitang/hvs-audit-20260921')
OLD=Path('/home/weitang/hvs-equivalence-20260921')
sys.path.insert(0,str(OLD))
from analyze import read_index
THREADS=int(subprocess.check_output(['nproc'],text=True))
env=dict(os.environ,OMP_NUM_THREADS=str(THREADS),OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
CASES=[('gaussian32',t) for t in [1,2,3,4]]+[('gaussian17',2),('sift128',1),('sift128',2),('glove100',1)]
INDEX_FILES=['index.bin','index2.bin','quantizer.gt','searching.gt']
(OUT/'raw').mkdir(exist_ok=True)
manifest={'upstream_revision':'f554268f8c6abcbc23ab35c3562703d7df8ae7b5','current_revision':subprocess.check_output(['git','-C',str(OUT.parents[1]),'rev-parse','HEAD'],text=True).strip(),'threads':THREADS,'numa':subprocess.check_output(['numactl','--interleave=all','numactl','--show'],text=True),'opencv_threads':1,'seed':100,'query_scheduling':'dynamic,1','data':{},'source_sha256':{}}
for p in (OUT.parents[1]/'hnsw').rglob('*'):
 if p.is_file():manifest['source_sha256'][str(p.relative_to(OUT.parents[1]))]=hashlib.sha256(p.read_bytes()).hexdigest()
(OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
def invoke(name,variant,mode,directory,data,meta,level):
 cmd=['numactl','--interleave=all',str(WORK/(variant+'-audit')),mode,str(data/'base.fvecs'),str(data/'query.fvecs'),str(meta['n']),str(meta['dim']),str(level),str(meta['nq'])]
 print('START',name,flush=True);start=time.monotonic()
 with (OUT/'raw'/f'{name}.log').open('w') as f:p=subprocess.run(cmd,cwd=directory,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=900)
 result={'command':cmd,'returncode':p.returncode,'threads':THREADS,'wall_seconds':time.monotonic()-start}
 (OUT/'raw'/f'{name}.json').write_text(json.dumps(result,indent=2)+'\n')
 if p.returncode:raise RuntimeError(name)
 assert f'AUDIT_THREADS {THREADS}' in (OUT/'raw'/f'{name}.log').read_text()
 print('DONE',name,round(result['wall_seconds'],2),flush=True)
def ids(directory):
 return {p.name:np.fromfile(p,dtype='<u4') for p in sorted(directory.glob('ids-k*-ef*.bin'))}
results=[]
for dataset,level in CASES:
 case=f'{dataset}-T{level}';data=OLD/'data'/dataset;meta=json.loads((data/'manifest.json').read_text())
 root=WORK/'runs'/case;current=root/'current';upstream=root/'upstream';reference=root/'upstream-on-current'
 for p in [current,upstream,reference]:p.mkdir(parents=True,exist_ok=False)
 invoke(case+'-current','current','build-query',current,data,meta,level)
 for name in INDEX_FILES:(reference/name).symlink_to(current/name)
 invoke(case+'-same-index-query','upstream','query',reference,data,meta,level)
 left=ids(current);same=ids(reference);assert left.keys()==same.keys() and len(left)==14
 compared=sum(v.size for v in left.values());different=sum(int(np.count_nonzero(v!=same[k])) for k,v in left.items())
 # Continue collecting independent-build evidence even if same-index queries differ.
 invoke(case+'-upstream-build','upstream','build',upstream,data,meta,level)
 invoke(case+'-upstream-query','upstream','query',upstream,data,meta,level)
 right=ids(upstream);a=read_index(upstream);b=read_index(current)
 training_keys=['rotation','start_merges','codebooks']+[f'layer{i}.merges' for i in range(1,level)]
 training={k:bool(np.array_equal(a['components'][k],b['components'][k])) for k in training_keys}
 index_fields={k:bool(np.array_equal(v,b['components'][k])) for k,v in a['components'].items()}
 gt=np.load(data/'groundtruth-ids.npy');query=[]
 for name,c in left.items():
  k,ef=map(int,name.removeprefix('ids-k').removesuffix('.bin').split('-ef'))
  u=right[name];uc=u.reshape(meta['nq'],k);cc=c.reshape(meta['nq'],k)
  ur=sum(len(set(x)&set(y)) for x,y in zip(uc,gt[:,:k]))/(meta['nq']*k)
  cr=sum(len(set(x)&set(y)) for x,y in zip(cc,gt[:,:k]))/(meta['nq']*k)
  query.append({'k':k,'ef':ef,'compared_ids':c.size,'same_index_different_ids':int(np.count_nonzero(c!=same[name])),'independent_build_different_ids':int(np.count_nonzero(c!=u)),'upstream_recall':ur,'current_recall':cr,'recall_delta_pp':100*(cr-ur)})
 row={'case':case,'threads':THREADS,'n':meta['n'],'nq':meta['nq'],'same_index_compared_ids':compared,'same_index_different_ids':different,'query_configurations':len(left),'scratch_reuse_checks':'17 query configurations per process, including a repeated ef=256 sweep; internal assertions passed','training_fields_equal':training,'independent_build_index_fields_equal':index_fields,'upstream_layer_counts':a['counts'],'current_layer_counts':b['counts'],'queries':query}
 results.append(row);(OUT/'audit-results.json').write_text(json.dumps(results,indent=2)+'\n')
 print('RESULT',case,'same-index differences',different,'training_equal',all(training.values()),'max_recall_delta_pp',max(abs(q['recall_delta_pp']) for q in query),flush=True)
assert len(results)==8 and all(r['same_index_different_ids']==0 for r in results)
print('AUDIT COMPLETE',sum(r['same_index_compared_ids'] for r in results),'same-index IDs compared',flush=True)
