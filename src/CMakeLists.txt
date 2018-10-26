set(SBENCH_LIB_SOURCES
    arguments.cpp
    blockgen.cpp
    except.cpp
    field_info.cpp
    field_repository.cpp
    result.cpp
    stencil_execution.cpp
    stencil_factory.cpp 
    table.cpp
    backend/util.cpp)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_BINARY_DIR}")

configure_file(config.h.in config.h)

add_executable(sbench main.cpp)
add_library(sbench-lib "${SBENCH_LIB_SOURCES}")
#target_compile_options(sbench PRIVATE "${OpenMP_CXX_FLAGS}")
#target_link_libraries(sbench PRIVATE "${OpenMP_CXX_FLAGS}")

if (SBENCH_WITH_PLATFORM_GENERIC)
    add_subdirectory(platform/generic)
    target_link_libraries(sbench PRIVATE sbench-platform-generic)
endif()

if (SBENCH_WITH_PLATFORM_KNL)
    add_subdirectory(platform/knl)
    target_link_libraries(sbench PRIVATE sbench-platform-knl)
endif()

if (SBENCH_WITH_PLATFORM_NVIDIA)
    add_subdirectory(platform/nvidia)
    target_link_libraries(sbench PRIVATE sbench-platform-nvidia)
endif()

target_link_libraries(sbench PRIVATE sbench-lib)
