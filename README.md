# HVS

HVS: Hierarchical Graph Structure Based on Voronoi Diagrams for Solving Approximate Nearest Neighbor Search

This fork of [Kejing-Lu/hvs](https://github.com/Kejing-Lu/hvs) provides an
in-memory library interface and reproducible source dependencies for
[artea-benchmark](https://github.com/NTU-Siqiang-Group/artea-benchmark).

## Getting Started

### Prerequisites

* Linux on x86-64, with a C/C++17 compiler supporting OpenMP
* CMake 3.24+
* Git, including recursively initialized submodules

OpenCV Core 4.5.4, Eigen 3.4.0 and zlib 1.3.2 are pinned Git submodules under `third-party/`.
CMake builds static libraries from these sources inside the build directory.
No system OpenCV, TBB, IPP, GUI libraries, or privileged installation is needed.
See [third-party/README.md](third-party/README.md) for dependency scope and build
details. OpenMP and the C/C++ runtimes come from the compiler toolchain.

### Datasets and query sets

* Tiny80M (https://drive.google.com/file/d/1PzW9cqi8VbzH9Bu_7UDWAXjA2DlBorG3/view?usp=sharing)
* Other real datasets (https://www.cse.cuhk.edu.hk/systems/hash/gqr/datasets.html)

### Compile on Linux

```shell
git clone --recurse-submodules git@github.com:WittenYeh/hvs.git
cd hvs
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j8
```

For an existing checkout, run `git submodule update --init --recursive` first.
HVS has one project-owned `CMakeLists.txt`, at the repository root. Dependency
submodules retain their own upstream build files. The demo is `build/main`.

To embed HVS, use `add_subdirectory(path/to/hvs)` and link `HVS::hvs`. Public
headers require C++17; OpenCV headers remain private to the implementation.
The first build compiles the pinned dependencies; subsequent builds reuse them.
## Commands

The benchmark integration also provides a standalone `HVS::hvs` CMake target
and `hnsw/hvs.hpp`. It builds an index in memory and searches it without accessing
index files:

```cpp
hvs::BuildConfig config;
hvs::Index index(base, num_vectors, dim, config);
const auto queue = std::min(std::size_t(100), index.max_search_queue());
hvs::Index::QueryScratch scratch(index, queue);
index.search(query, topk, queue, ids, scratch);
```

Use one scratch object per query worker, with `0 < topk <= queue`. The library
copies input vectors during construction. `quantizer.cpp` contains the training
routines extracted from the original demo; both entry points share those
implementations. Training retains the original initialization, iteration counts,
merging and representative-selection rules, while using scoped OpenCV storage,
partial candidate sorting, reusable query buffers and parallel routing cells.
The graph releases its own quantized layers and can finish construction without
serializing the original-vector graph.

The original file-based demo below is built when `HVS_BUILD_DEMO=ON` (the default
for a standalone build). It is disabled by default when included as a subdirectory.

* `K` is the value of top-K, `L` is the value of efsearch and `qn` is the size of query set
* `T` and `delta` are user-specified parameters of HVS
* The data set, query set and the ground_truth set are stored in dPath.ds, qPath.q and truth.gt

Build HVS index
```shell
./build/main ${dPath}.ds nullptr ${n} ${d} ${T} -1 ${delta} -1
```
Search in HVS
```shell
./build/main ${dPath}.ds ${qPath}.q ${n} ${d} ${T} ${qn} ${K} ${L}
```

## A running example (ImageNet)
* Donwload the dataset, query set and the ground truth set of ImageNet from the following link
https://drive.google.com/file/d/1WV78sZT1j1oz-GoZTQdyKaNQgulRLe2O/view?usp=sharing
* Put all three files in the main folder
* Run the script
```shell
bash run_image.sh
```
