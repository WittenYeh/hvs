"""Prepare isolated source snapshots and compile the correctness harnesses."""
from pathlib import Path
import io
import json
import shutil
import subprocess
import tarfile

OUT = Path(__file__).resolve().parent
HVS = OUT.parents[1]
REPO = OUT.parents[3]
WORK = Path('/home/weitang/hvs-audit-20260921')
WORK.mkdir(exist_ok=True)
revision = 'f554268f8c6abcbc23ab35c3562703d7df8ae7b5'
archive = subprocess.check_output(['git', '-C', str(HVS), 'archive', revision])
with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
    tar.extractall(WORK / 'upstream', filter='data')
shutil.copytree(HVS / 'hnsw', WORK / 'current/hnsw', dirs_exist_ok=True)
header = WORK / 'current/hnsw/hvs.hpp'
source = header.read_text()
assert source.count('class Index {') == 1
header.write_text(source.replace('class Index {', 'class Index {\n  public: // Audit observer access only.'))
for name in ['highgui/highgui.hpp', 'imgproc/imgproc.hpp']:
    path = WORK / 'compat/opencv2' / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('#pragma once\n#include <opencv2/core/core_c.h>\n')
build = REPO / 'bench-hvs/build'
flags = ['/home/weitang/.local/bin/g++', '-std=c++20', '-O3', '-DNDEBUG',
         '-march=native', '-fopenmp', '-include', 'algorithm', '-include',
         'opencv2/core/core_c.h', '-I' + str(WORK / 'compat'),
         '-I' + str(HVS / 'third-party/opencv/modules/core/include'),
         '-I' + str(build), '-I' + str(OUT)]
libraries = [str(build / 'hvs/third-party/opencv/lib/libopencv_core.a'),
             str(build / 'hvs/third-party/zlib/libz.a'), '-lpthread', '-ldl', '-lrt']
commands = {
    'current': flags + ['-I' + str(WORK / 'current/hnsw'), str(OUT / 'query_audit.cpp'),
                        str(WORK / 'current/hnsw/quantizer.cpp')] + libraries +
                       ['-o', str(WORK / 'current-audit')],
    'upstream': flags + ['-DORIGINAL', '-I' + str(WORK / 'upstream/hnsw'),
                         str(OUT / 'query_audit.cpp'), '-Wl,--wrap=time'] + libraries +
                        ['-o', str(WORK / 'upstream-audit')],
}
(OUT / 'compile-commands.json').write_text(json.dumps(commands, indent=2) + '\n')
for name, command in commands.items():
    with (WORK / f'compile-{name}.log').open('w') as log:
        subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
    print('Compiled', name, flush=True)
