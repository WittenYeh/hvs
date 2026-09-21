# HVS performance comparison on SIFT1M

This directory contains a controlled comparison of the current HVS library and
original `Kejing-Lu/hvs` revision `f554268f8c6abcbc23ab35c3562703d7df8ae7b5`.
The current revision and exact source/library/data fingerprints are recorded in
`manifest.json` and `compile-commands.json`.

The [repository README](../README.md#implementation-validation-and-performance)
embeds the recall/QPS, recall-difference and build-time plots with a concise
interpretation. The [implementation audit](implementation-audit/README.md)
contains the separate 64-thread correctness checks.

**Historical performance settings:** this run used 16 physical cores on one
NUMA socket without interleave. It predates the current policy of all available
OpenMP threads (`nproc`) and `numactl --interleave=all`. The archived driver
retains the measured settings for provenance; it is not the recommended setup
for a new performance run. No 64-thread performance results are claimed here.

## Workload and controls

- Full SIFT1M: 1,000,000 base vectors, all 10,000 queries, 128 dimensions.
- Supplied full-dataset SIFT ground truth, at least 100 neighbors per query.
- T=1, M=16, efConstruction=500, delta=0.5, 10,000 training samples, seed=100.
- Both use GCC 14.3, C++20, `-O3 -DNDEBUG -march=native
  -mprefer-vector-width=512 -fopenmp`, and exactly the same static OpenCV Core
  4.5.4 library (Eigen 3.4.0, zlib 1.3.2).
- This uses common optimization flags to isolate implementation differences;
  the upstream CMake default `-Ofast` is not used in the primary comparison.
- Construction: 16 physical CPU cores, CPU affinity 0–15 on one NUMA socket of
  an Intel Xeon Gold 6326. OpenCV is limited to one thread in both versions.
- Query throughput: 1 and 16 threads, one independent scratch area per worker,
  static OpenMP scheduling, same affinity and same query order.
- One build per variant, as requested. Original runs first, then current. Variants run sequentially, with no overlapping benchmark runs.
- k=1,10,100; ef=1,2,4,8,12,16,24,32,48,64,96,128,192,256,384,512, retaining ef>=k.
- Each configuration warms 1,000 queries and then times all 10,000 queries.
  Each timed sample executes complete query batches until at least 0.12 seconds.
  QPS is total queries / measured wall time. Each point is one measurement on that variant’s single index; no confidence
  interval or run-to-run variability is estimated.
- Recall@k is the mean ID-set intersection with the top-k ground truth divided
  by k. It is the same definition for both variants, including k=1; it is not
  the benchmark suite's distance-based Soft Recall@1.

## Timing boundaries

The primary construction measure is **time until an index is query-ready**,
with the dataset file cache warmed for both variants. For the current library,
this is base-vector loading plus the `hvs::Index` constructor. For the original,
it is the full native `sift_test1B` construction call (including its required
file reads/writes) plus loading the index and query metadata in a fresh process.
Process startup, query-vector loading, ground-truth loading and query execution
are excluded. Original construction and query use separate processes, as in
its normal workflow, so leaked training allocations do not remain resident
while timing queries.

`build_times.csv` also exposes the separate API time, the original demo's own
internal construction timer and loading times. The original internal timer
excludes some preparation/final writes but includes its base-graph write; it
should not be interpreted as an identical boundary to the current constructor.
The API/ready-time comparison measures the practical difference between the
original file-based integration and the current in-memory integration, including
that architectural change; it is not a pure CPU-kernel-only build comparison.
Both API totals include quantizer training and full-vector encoding.

Queries include copying/padding the input, rotation, distance-table construction,
routing and all graph traversal. They exclude index loading, one-time scratch
allocation, recall evaluation and result-file writing. The original demo prints
a traversal-only query timer after precomputing distance tables. We do not use
that narrower number as the original's QPS in these plots.

`original_adapter.hpp` places the original demo's complete per-query steps behind
the same timing loop. It calls upstream `restore_index`, `restore_index2` and
`SearchWith*Graph` functions directly; their implementations, including native
per-call candidate allocation, are preserved. Only worker scratch ownership and
the timing harness are adapted to permit the same multithreaded driver.
This means the 16-thread upstream curve measures concurrent original library
queries through this adapter; the upstream demo itself is serial.

The current runner links the actual benchmark's `libhvs.a`. The original is
extracted with `git archive`; its only source patch stores the existing native
build timer in an observer variable. Unused GUI includes use Core-only shim
headers. `--wrap=time` makes the native `srand(time(NULL))` reproducible.

`adapter-verification.json` records a check against saved, directly executed
original-demo results: all 14 k/ef configurations on 10,000 SIFT base vectors,
257,000 returned IDs per implementation, identical ordered results. Parallel
construction can change insertion order even with a fixed seed, so two full-scale
parallel builds are not expected to have bitwise-identical indices. This single
build comparison does not quantify variation between independent builds.

## Outputs

- `recall_vs_qps.png`, `.pdf`, `.svg`: six panels (k=1/10/100 × 1/16 threads).
- `build_time.png`, `.pdf`, `.svg`: construction API and query-ready wall time.
- `recall_difference.png`, `.pdf`, `.svg`, `.csv`: signed recall differences
  (current minus original, in percentage points) at the same k/ef settings.
- `query_samples.csv`: every timed sample, including batch count and duration.
- `query_by_build.csv`: per-build QPS and recall.
- `recall_vs_qps.csv`: plotted measurements (median/min/max fields coincide for one build).
- `build_times.csv`: individual construction and loading measurements.
- `matched_recall.csv`: QPS at recall targets 0.90/0.95/0.99, obtained by
  interpolation in log-QPS between measured curve points. These are
  interpolated comparisons, not additional measured parameter settings; targets
  outside either curve's measured range are omitted.
- `summary.json`, `REPORT.md`: machine-readable and human-readable summaries.
- `raw/`: native stdout, per-run commands, exit status and process wall times.
  `raw/diagnostics/` preserves the interrupted preliminary query sweep with
  three timing samples; it is excluded from all reported measurements.

Large native index artifacts and build executables live outside the repository
in `/home/weitang/hvs-performance-20260921`. Benchmarking does not edit HVS's
algorithm source or add another CMake entry point.

## Regenerate plots and reports

Only Python, NumPy and Matplotlib are required; these commands use the archived
measurements without rebuilding an index or running any queries:

```sh
python3 results/plot_results.py
python3 results/plot_differences.py
python3 results/write_report.py
python3 results/implementation-audit/write_report.py
```

Run these commands from the HVS repository root. `plot_differences.py` also
exports the audit recall-difference chart and its CSV data.

## Archived benchmark reproduction on the original machine

From the artea-benchmark root, first build `./scripts/install.sh --bench hvs`.
The commands below reproduce the historical 16-core configuration. For new
measurements, adjust the driver to use all available OpenMP threads and NUMA
interleave, and keep the new outputs separate from this archive:

```sh
python3 third-party/hvs/results/prepare_benchmark.py
python3 third-party/hvs/results/verify_adapter.py
python3 third-party/hvs/results/run_benchmark.py
/home/weitang/miniconda3/envs/depot/bin/python third-party/hvs/results/plot_results.py
python3 third-party/hvs/results/write_report.py
```

The verification script uses the previous equivalence fixtures under
`/home/weitang/hvs-equivalence-20260921`. The performance run uses full SIFT1M
files already present in `third-party/ann-datasets/sift-1m`. Existing run
directories are protected against overwrite; `--resume` reuses completed stages; use `--start-repeat 2 --repeats 1`
for another set and choose a separate results directory when comparing batches.
Absolute paths, CPU affinity and compiler paths in these scripts describe the
measured machine and need adjustment on a different machine.
