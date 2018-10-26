set(SBENCH_PLATFORM_NVIDIA_FLAGS "" CACHE STRING "NVIDIA platform compiler flags")

add_library(sbench-platform-nvidia
            nvidia.cu
            ../../backend/cuda/hdiff_otf.cu
            ../../backend/cuda/hdiff_naive.cu)
set_target_properties(sbench-platform-nvidia PROPERTIES CUDA_STANDARD 11)
target_compile_options(sbench-platform-nvidia PRIVATE "SHELL:${SBENCH_PLATFORM_NVIDIA_FLAGS}")
target_compile_options(sbench-platform-nvidia PRIVATE -gencode=arch=compute_35,code=sm_35)
target_compile_options(sbench-platform-nvidia PRIVATE -gencode=arch=compute_37,code=sm_37)
target_compile_options(sbench-platform-nvidia PRIVATE -gencode=arch=compute_60,code=sm_60)
target_compile_options(sbench-platform-nvidia PRIVATE -gencode=arch=compute_70,code=sm_70)
target_link_libraries(sbench-platform-nvidia PRIVATE sbench-lib)
