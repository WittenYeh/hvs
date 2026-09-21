"""Run paired SIFT1M benchmarks sequentially on 16 physical cores of socket 0."""
from pathlib import Path
import argparse, hashlib, json, os, struct, subprocess, time
ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[2]
WORK=Path('/home/weitang/hvs-performance-20260921')
DATA=REPO/'third-party/ann-datasets/sift-1m'
parser=argparse.ArgumentParser()
parser.add_argument('--repeats',type=int,default=1)
parser.add_argument('--start-repeat',type=int,default=1)
parser.add_argument('--resume',action='store_true')
a=parser.parse_args()
base=DATA/'sift_base.fvecs';query=DATA/'sift_query.fvecs';truth=DATA/'sift_groundtruth.ivecs'
def shape(p):
 with p.open('rb') as f:d=struct.unpack('<i',f.read(4))[0]
 assert p.stat().st_size%(4*(d+1))==0
 return p.stat().st_size//(4*(d+1)),d
assert shape(base)==(1000000,128)
assert shape(query)==(10000,128)
assert shape(truth)[0]==10000 and shape(truth)[1]>=100
manifest={'dataset':'SIFT1M','n':1000000,'dimension':128,'nq':10000,'ground_truth':str(truth),
 'current_commit':subprocess.check_output(['git','-C',str(ROOT.parent),'rev-parse','HEAD'],text=True).strip(),
 'upstream_commit':'f554268f8c6abcbc23ab35c3562703d7df8ae7b5',
 'parameters':{'T':1,'M':16,'efConstruction':500,'delta':0.5,'training_samples':10000,'seed':100},
 'build_threads':16,'query_threads':[1,16],'opencv_threads':1,'cpus':'0-15','physical_cores':16,
 'independent_builds_per_variant':a.repeats,'first_repeat':a.start_repeat,'warmup_queries_per_query_configuration':1000,
 'timing_samples_per_build':1,'minimum_seconds_per_timing_sample':0.12,
 'recall_definition':'ID intersection with full SIFT ground truth, divided by nq*k',
 'query_timing':'query copy/padding + rotation + distance tables + routing + all graph traversal; excludes setup, index loading, GT evaluation and result serialization',
 'build_timing':'warm file cache to query-ready index: current data load + constructor; upstream native build API including its I/O + index/metadata reload in a fresh process',
 'cpu':subprocess.check_output(['lscpu'],text=True),
 'compiler':subprocess.check_output(['/home/weitang/.local/bin/g++','--version'],text=True).splitlines()[0],
 'library_sha256':{},'data_sha256':{}}
for p in [base,query,truth]:
 h=hashlib.sha256()
 with p.open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 manifest['data_sha256'][str(p)]=h.hexdigest()
for p in [REPO/'bench-hvs/build/hvs/libhvs.a',REPO/'bench-hvs/build/hvs/third-party/opencv/lib/libopencv_core.a',REPO/'bench-hvs/build/hvs/third-party/zlib/libz.a']:
 manifest['library_sha256'][str(p)]=hashlib.sha256(p.read_bytes()).hexdigest()
manifest['runner_source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [ROOT/'benchmark.cpp',ROOT/'original_adapter.hpp']}
manifest['upstream_source_patch_sha256']=hashlib.sha256((ROOT/'upstream-timing.patch').read_bytes()).hexdigest()
manifest['submodules']={name:subprocess.check_output(['git','-C',str(ROOT.parent/'third-party'/name),'rev-parse','HEAD'],text=True).strip() for name in ['opencv','eigen','zlib']}
(ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
env=dict(os.environ,OMP_NUM_THREADS='16',OMP_PLACES='cores',OMP_PROC_BIND='close',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENCV_FOR_THREADS_NUM='1')
logs=ROOT/'raw';logs.mkdir(exist_ok=True)
def invoke(variant,mode,directory,label,repeat):
 path=logs/(label+'.log')
 meta_path=logs/(label+'.json')
 if a.resume and meta_path.exists() and json.loads(meta_path.read_text())['returncode']==0:
  print('REUSE',label,flush=True);return
 cmd=['taskset','-c','0-15',str(WORK/(variant+'-benchmark')),mode,str(base),str(query),str(truth),'1000000','128','1','16']
 print('START',label,time.strftime('%H:%M:%S'),flush=True)
 start=time.monotonic()
 with path.open('w') as out:
  p=subprocess.run(cmd,cwd=directory,env=env,stdout=out,stderr=subprocess.STDOUT)
 meta={'repeat':repeat,'variant':variant,'mode':mode,'command':cmd,'returncode':p.returncode,'wall_seconds':time.monotonic()-start,'load_average':os.getloadavg()}
 (logs/(label+'.json')).write_text(json.dumps(meta,indent=2)+'\n')
 if p.returncode:raise RuntimeError(f'{label} failed: see {path}')
 print('DONE',label,round(meta['wall_seconds'],2),flush=True)
for repeat in range(a.start_repeat,a.start_repeat+a.repeats):
 for variant in (['upstream','current'] if repeat%2 else ['current','upstream']):
  directory=WORK/'runs'/f'r{repeat}-{variant}'
  directory.mkdir(parents=True,exist_ok=a.resume)
  # Warm the base file cache equally before either API; no cache dropping/root access.
  with base.open('rb') as f:
   while f.read(8*1024*1024):pass
  if variant=='upstream':
   invoke(variant,'build',directory,f'r{repeat}-{variant}-build',repeat)
   invoke(variant,'query',directory,f'r{repeat}-{variant}-query',repeat)
  else:invoke(variant,'bench',directory,f'r{repeat}-{variant}-bench',repeat)
print('ALL RUNS COMPLETE',flush=True)
