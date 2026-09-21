# HVS implementation audit

This audit compares current HVS commit `33de94f523fa46fc8703411148b50fc89ada3f8c`
with the original GitHub repository's current main/HEAD,
`f554268f8c6abcbc23ab35c3562703d7df8ae7b5`, verified with `git ls-remote`.
It is a correctness and source review, not a performance measurement.

![Maximum absolute recall differences across the eight audit cases](recall_difference.png)

The orange bars show each case's maximum absolute recall difference across
k/ef settings after independent parallel builds. The teal markers show zero
recall difference when both query implementations use exactly the same index:
all 974,544 ordered result IDs match across 112 configurations. The bars are
maxima from one build per variant, not confidence bounds. Per-case plotted
values are in [recall_difference.csv](recall_difference.csv); signed per-query
differences are in [audit-results.json](audit-results.json).

Regenerate PNG/PDF/SVG and CSV exports from the HVS repository root with
`python3 results/plot_differences.py` (NumPy and Matplotlib must be installed).
The separate full SIFT1M performance and construction-time plots are embedded
in the [repository README](../../README.md#implementation-validation-and-performance).

## Source checks

`inspect_sources.py` extracts function bodies from the current and exact archived
upstream `hnswalg.h`. Fourteen groups of graph functions (including overloaded
insertion and distance functions) are identical after removing comments and
whitespace. Five query traversal functions are identical after additionally
normalizing only local-vs-reusable candidate storage and removing an unused
original vector. Arithmetic, comparisons, traversal order and ID mapping are
not normalized. Five other graph headers are byte-identical to upstream.

Every current algorithm source/header is SHA-256 identical to the version used
in the previous controlled equivalence test. `historical-validation-summary.json`
contains the previous final dependency-validation results, explicitly labeled
historical; those single-thread builds were not rerun in this audit.

The orchestration and training rewrite is reviewed separately, including PCA/OPQ,
codebook pairing/merging, representative selection, layer transitions and query
lookup-table generation. Function-body equality does not apply to that rewrite.

## New runtime checks

All newly executed index builds and query batches use the system's maximum
available OpenMP thread count (`nproc`, 64 on this machine) and are launched with
`numactl --interleave=all`. No CPU subset is pinned. OpenCV training is fixed at
one internal thread in both variants to hold the numerical backend constant.
`AUDIT_THREADS` in each process log verifies the actual OpenMP team size.

There are eight cases, each with 10,000 base vectors:
Gaussian32 T=1/2/3/4, Gaussian17 T=2, SIFT128 T=1/2, and GloVe100 T=1.
The previous fixtures supply 256, 256, 1,000 and 512 queries respectively, and
float64 exact ground truth for each 10,000-vector subset.

Each case runs sequentially:

1. Build the current index with 64 threads, export its state, and query it with
   64 threads. An isolated copy of `hvs.hpp` exposes `Index::impl_` solely for
   post-construction observation/serialization; production files are untouched.
2. Load exactly that exported index with the original library and run the
   original query functions using 64 threads through `original_adapter.hpp`.
   Compare every ordered returned ID to step 1. This holds the graph, codebooks,
   mappings and entry routes fixed, isolating query behavior from build order.
3. Independently build the original index with 64 threads, and query it with
   the same driver. Compare training state, live index fields and recall to
   step 1. Parallel graphs are allowed to differ; differences are reported.

Both variants use GCC 14.3, C++20, `-O3 -DNDEBUG -march=native -fopenmp`, and the
same static OpenCV/Eigen/zlib build. Training count=10,000, M=16,
efConstruction=500, delta=0.5, graph/C/OpenCV seeds=100. The original
`srand(time(NULL))` is controlled with linker `--wrap=time`, not a source change.
Original algorithm sources are extracted without patches using `git archive`.

Queries use dynamic OpenMP scheduling and per-worker scratch. Queue capacities
are swept non-monotonically: 256, 1, 32, 8, 128, 16, 64, then 256 again. For each
queue, k=1/10/100 is retained when k<=queue. The repeated final sweep must exactly
match its first execution, testing scratch reuse. Each process checks IDs are
in range and unique per result row. Fourteen unique settings per case are saved
for cross-implementation comparison; 17 settings execute including repeats.

The upstream adapter calls native distance-table and graph traversal functions.
It supplies thread-local scratch and a common input/output interface; the native
upstream demo itself queries serially. The adapter was previously checked
against directly executed upstream-demo ordered IDs; this audit extends its
use across all four supported T values.

## Reproduce on this machine

The benchmark's static dependencies must already be built. The previous fixture
and analysis directory `/home/weitang/hvs-equivalence-20260921` is required.
Large index snapshots and executables are kept outside the repository in
`/home/weitang/hvs-audit-20260921`.

```sh
python3 third-party/hvs/results/implementation-audit/prepare_audit.py
python3 third-party/hvs/results/implementation-audit/inspect_sources.py
/home/weitang/miniconda3/envs/depot/bin/python third-party/hvs/results/implementation-audit/run_audit.py
```

Existing run directories cause an error instead of being overwritten. Choose a
new WORK path in the scripts to repeat the experiment. Per-process raw logs,
commands and return codes are under `raw/`; exact results are in
`audit-results.json`. `manifest.json` records the machine policy, source revision
and fingerprints; `source-comparison.json` records mechanical source checks.
See `REPORT.md` for the reviewed conclusions and limitations.
