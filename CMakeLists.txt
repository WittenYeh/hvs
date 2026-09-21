cmake_minimum_required(VERSION 3.24)
project(HVS LANGUAGES C CXX)

find_package(OpenMP REQUIRED)
option(HVS_BUILD_DEMO "Build the original file-based demo" ${PROJECT_IS_TOP_LEVEL})

foreach(dependency IN ITEMS opencv eigen zlib)
    if(NOT EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third-party/${dependency}/CMakeLists.txt")
        message(FATAL_ERROR "Missing ${dependency}: run git submodule update --init --recursive")
    endif()
endforeach()

# Scope dependency options so they do not change the including project's settings.
function(hvs_add_dependencies)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    set(CMAKE_POLICY_DEFAULT_CMP0126 NEW)
    set(CMAKE_CXX_STANDARD 11)
    set(BUILD_SHARED_LIBS OFF)
    set(ZLIB_BUILD_SHARED OFF)
    set(ZLIB_BUILD_TESTING OFF)
    add_subdirectory(third-party/zlib EXCLUDE_FROM_ALL)
    export(EXPORT zlibStaticExport FILE "${CMAKE_CURRENT_BINARY_DIR}/ZlibTargets.cmake")

    set(BUILD_LIST core)
    set(ZLIB_LIBRARY zlibstatic)
    set(ZLIB_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third-party/zlib")
    set(WITH_EIGEN ON)
    set(CMAKE_DISABLE_FIND_PACKAGE_Eigen3 ON)
    set(EIGEN_INCLUDE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/third-party/eigen")
    set(OPENCV_CONFIG_FILE_INCLUDE_DIR "${CMAKE_BINARY_DIR}")
    # Only Core, Eigen and zlib are needed. Disable optional libraries/downloads.
    foreach(option IN ITEMS
            BUILD_ZLIB BUILD_TESTS BUILD_PERF_TESTS BUILD_EXAMPLES BUILD_JAVA
            BUILD_opencv_apps BUILD_opencv_python2 BUILD_opencv_python3 BUILD_opencv_gapi
            WITH_IPP WITH_ITT WITH_TBB WITH_LAPACK WITH_OPENCL WITH_CUDA
            WITH_VA WITH_VA_INTEL WITH_PROTOBUF WITH_QUIRC
            WITH_JPEG WITH_PNG WITH_TIFF WITH_WEBP WITH_OPENEXR WITH_OPENJPEG WITH_JASPER
            WITH_FFMPEG WITH_GSTREAMER WITH_GTK WITH_V4L WITH_VTK PARALLEL_ENABLE_PLUGINS)
        set(${option} OFF)
    endforeach()
    add_subdirectory(third-party/opencv EXCLUDE_FROM_ALL)
endfunction()
hvs_add_dependencies()

add_library(hvs STATIC hnsw/hvs.cpp hnsw/quantizer.cpp)
add_library(HVS::hvs ALIAS hvs)
target_compile_features(hvs PUBLIC cxx_std_17)
target_compile_options(hvs PRIVATE -march=native)
target_include_directories(hvs PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/hnsw")
target_include_directories(hvs SYSTEM PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv/modules/core/include"
    "${CMAKE_BINARY_DIR}")
target_link_libraries(hvs PRIVATE opencv_core OpenMP::OpenMP_CXX)

if(HVS_BUILD_DEMO)
    add_executable(main hnsw/main.cpp hnsw/sift_1b.cpp)
    set_target_properties(main PROPERTIES RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")
    target_compile_options(main PRIVATE -march=native)
    target_include_directories(main SYSTEM PRIVATE
        "${CMAKE_CURRENT_SOURCE_DIR}/third-party/opencv/modules/core/include"
        "${CMAKE_BINARY_DIR}")
    target_link_libraries(main PRIVATE HVS::hvs OpenMP::OpenMP_CXX)
endif()
