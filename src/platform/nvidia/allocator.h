#pragma once

#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

#include "except.h"

namespace platform {
    namespace nvidia {

        template <class ValueType>
        struct cuda_allocator {
            using value_type = ValueType;

            template <class OtherValueType>
            struct rebind {
                using other = cuda_allocator<OtherValueType>;
            };

            value_type *allocate(std::size_t n) const {
                value_type *ptr;
#ifdef __CUDACC__
                if (cudaMallocManaged(reinterpret_cast<void **>(&ptr), n * sizeof(value_type)) != cudaSuccess)
                    throw ERROR("could not allocate managed memory");
#else
                throw ERROR("compiled without CUDA support");
#endif
                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) const {
#ifdef __CUDACC__
                if (cudaFree(ptr) != cudaSuccess)
                    throw ERROR("could not free managed memory");
#endif
            }
        };
    } // namespace nvidia
} // namespace platform
