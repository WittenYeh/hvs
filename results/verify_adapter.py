from pathlib import Path
import json, os, subprocess
ROOT=Path(__file__).resolve().parent
WORK=Path('/home/weitang/hvs-performance-20260921')
OLD=Path('/home/weitang/hvs-equivalence-20260921')
DATA=OLD/'data/sift128'
env=dict(os.environ,OMP_NUM_THREADS='1',OMP_PLACES='cores',OMP_PROC_BIND='close',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
reports=[]
for variant,mode in [('upstream','query'),('current','bench')]:
 directory=WORK/'verify'/variant;directory.mkdir(parents=True,exist_ok=True)
 reference=OLD/'runs/sift128-T1-omp1-r1/upstream'
 if variant=='upstream':
  for name in ['index.bin','index2.bin','quantizer.gt','searching.gt']:
   p=directory/name
   if not p.exists():p.symlink_to(reference/name)
 cmd=['taskset','-c','0',str(WORK/(variant+'-benchmark')),mode,str(DATA/'base.fvecs'),str(DATA/'query.fvecs'),str(DATA/'truth.gt'),'10000','128','1','1','verify']
 with (directory/'run.log').open('w') as out:subprocess.run(cmd,cwd=directory,env=env,stdout=out,stderr=subprocess.STDOUT,check=True)
 files=sorted(directory.glob('ids-k*-ef*.bin'))
 assert len(files)==14
 assert all(p.read_bytes()==(reference/p.name).read_bytes() for p in files),'Adapter changed result IDs'
 reports.append({'variant':variant,'query_configurations':len(files),'compared_ids':sum(p.stat().st_size//4 for p in files),'different_ids':0})
 print(reports[-1],flush=True)
(ROOT/'adapter-verification.json').write_text(json.dumps(reports,indent=2)+'\n')
