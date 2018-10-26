set(SBENCH_PLATFORM_GENERIC_FLAGS "" CACHE STRING "Generic platform compiler flags")

add_library(sbench-platform-generic generic.cpp)
target_compile_options(sbench-platform-generic PRIVATE "SHELL:${SBENCH_PLATFORM_GENERIC_FLAGS}")
target_link_libraries(sbench-platform-generic PRIVATE sbench-lib OpenMP::OpenMP_CXX)
