cmake_minimum_required(VERSION 3.9)

project(sbench VERSION 0.0.0)
enable_language(CXX)

# Build-time configuration

set(SBENCH_FLOAT_TYPE double CACHE STRING "Floating point type")
set_property(CACHE SBENCH_FLOAT_TYPE PROPERTY STRINGS "float" "double")

option(SBENCH_WITH_PLATFORM_GENERIC "Build generic platform" ON)
option(SBENCH_WITH_PLATFORM_KNL "Build KNL platform" OFF)
option(SBENCH_WITH_PLATFORM_NVIDIA "Build NVIDIA platform" OFF)

if (NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release")
endif()

if (SBENCH_WITH_PLATFORM_NVIDIA)
    enable_language(CUDA)
    set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
endif()

set(CMAKE_CXX_STANDARD 11)

find_package(OpenMP REQUIRED)

add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_DEBUG}>"
    "$<$<COMPILE_LANGUAGE:CXX>:${CXXOPT_WALL}>"
    )
add_subdirectory(src)
