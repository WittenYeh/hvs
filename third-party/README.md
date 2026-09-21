# HVS source dependencies

Initialize all dependencies with `git submodule update --init --recursive`.
The gitlinks pin exact source commits; builds do not follow upstream branches.

| Submodule | Release | Commit | Use |
| --- | --- | --- | --- |
| [opencv](https://github.com/opencv/opencv) | 4.5.4 | `4223495e6cd67011f86b8ecd9be1fa105018f3b1` | Core matrices, PCA, SVD and matrix products used by quantizer training. |
| [eigen](https://github.com/eigen-mirror/eigen) | 3.4.0 | `3147391d946bb4b6c68edd901f2add6ac1f31f8c` | Header-only linear algebra backend used by OpenCV's eigenvalue decomposition, matching the previous OpenCV package. |
| [zlib](https://github.com/madler/zlib) | 1.3.2 | `da607da739fa6047df13e66a2af6b8bec7c2a498` | OpenCV Core's required compression dependency. |

The root `CMakeLists.txt` adds OpenCV and zlib directly to the build as static
libraries and points OpenCV at the Eigen submodule's headers. No installation
step or separate dependency build driver is needed. OpenCV
uses this zlib submodule, not a host zlib or OpenCV's bundled zlib copy.
Only OpenCV Core is built. Optional dependencies such as IPP, ITT, TBB, LAPACK,
OpenCL, image codecs, GUI libraries, Python bindings and plugins are
disabled; dependency configuration does not download additional packages.
OpenCV's internal source files and their license notices remain in the pinned
OpenCV repository. Component licenses are in the respective submodules.

The compiler supplies OpenMP and the C/C++ runtimes, and the operating system
supplies pthreads and the dynamic-loader interface. These are toolchain/system
prerequisites, not separately fetched project libraries.
`hnsw/hnswlib` contains HVS's modified graph algorithm and is part of HVS itself;
replacing it with vanilla nmslib/hnswlib would remove the quantized hierarchy.

Use the usual `cmake --build build -j8` to control parallel compilation.
Dependencies follow the enclosing project's build type and compiler settings;
OpenCV retains its default CPU dispatch. `HVS_BUILD_DEMO` is the only HVS-specific
build option, on by default for standalone builds and off when embedded.
The old `HVS_OPENCV_ROOT` variable and home-directory OpenCV lookup are no
longer used. Changing the compiler, dependency revision or dependency build
options should use a fresh build directory.

OpenCV 4.5.4 preserves the source release used in the original integration.
Numerical results can still depend on compiler flags and enabled OpenCV
backends; a source-built library should be validated before reusing results
obtained with a differently configured system OpenCV package.
