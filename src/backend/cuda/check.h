#pragma once

#include "except.h"
#include <string>

inline void cuda_check(cudaError_t err, const std::string& code, const std::string& file, int line) {
    if (err != cudaSuccess) {
        throw error(file, line, cudaGetErrorString(err) + (" in " + code));
    }
}

#define CUDA_CHECK(code) cuda_check(code, #code, __FILE__, __LINE__)
