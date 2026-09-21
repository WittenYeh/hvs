"""Extract the original HVS revision and compile both variants identically.
Run the parent benchmark build first: ./scripts/install.sh --bench hvs
"""
from pathlib import Path
import difflib, io, json, subprocess, tarfile
ROOT=Path(__file__).resolve().parent
HVS=ROOT.parent
REPO=ROOT.parents[2]
WORK=Path('/home/weitang/hvs-performance-20260921')
BUILD=REPO/'bench-hvs/build'
WORK.mkdir(exist_ok=True)
revision='f554268f8c6abcbc23ab35c3562703d7df8ae7b5'
archive=subprocess.check_output(['git','-C',str(HVS),'archive',revision])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:tar.extractall(WORK/'upstream',filter='data')
p=WORK/'upstream/hnsw/sift_1b.cpp';before=p.read_text()
anchor='    cout << "Build time:" << 1e-6 * stopw_full.getElapsedTimeMicro() << "  seconds\\n";'
assert before.count(anchor)==1
new=before.replace(anchor,'    perf_native_build_seconds = 1e-6 * stopw_full.getElapsedTimeMicro();\n'+anchor)
p.write_text(new)
(ROOT/'upstream-timing.patch').write_text(''.join(difflib.unified_diff(before.splitlines(True),new.splitlines(True),fromfile='a/hnsw/sift_1b.cpp',tofile='b/hnsw/sift_1b.cpp')))
for header in ['highgui/highgui.hpp','imgproc/imgproc.hpp']:
 p=WORK/'compat/opencv2'/header;p.parent.mkdir(parents=True,exist_ok=True);p.write_text('#pragma once\n#include <opencv2/core/core_c.h>\n')
common=['/home/weitang/.local/bin/g++','-std=c++20','-O3','-DNDEBUG','-march=native','-mprefer-vector-width=512','-fopenmp','-include','algorithm','-include','opencv2/core/core_c.h','-I'+str(WORK/'compat'),'-I'+str(HVS/'third-party/opencv/modules/core/include'),'-I'+str(BUILD),'-I'+str(ROOT)]
link=[str(BUILD/'hvs/third-party/opencv/lib/libopencv_core.a'),str(BUILD/'hvs/third-party/zlib/libz.a'),'-lpthread','-ldl','-lrt']
commands={
 'upstream':[*common,'-DORIGINAL','-I'+str(WORK/'upstream/hnsw'),str(ROOT/'benchmark.cpp'),'-Wl,--wrap=time',*link,'-o',str(WORK/'upstream-benchmark')],
 'current':[*common,'-I'+str(HVS/'hnsw'),str(ROOT/'benchmark.cpp'),str(BUILD/'hvs/libhvs.a'),'-Wl,--wrap=time',*link,'-o',str(WORK/'current-benchmark')]}
(ROOT/'compile-commands.json').write_text(json.dumps(commands,indent=2)+'\n')
for name,cmd in commands.items():
 with (WORK/f'compile-{name}.log').open('w') as log:subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,check=True)
 print('Compiled',name,flush=True)
