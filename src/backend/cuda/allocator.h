#pragma once

#include "except.h"

namespace backend {
    namespace cuda {

        template <class ValueType>
        struct allocator {
            using value_type = ValueType;

            template <class OtherValueType>
            struct rebind {
                using other = allocator<OtherValueType>;
            };

            value_type *allocate(std::size_t n) const {
                value_type *ptr;
                if (cudaMallocManaged(reinterpret_cast<void **>(&ptr), n * sizeof(value_type)) != cudaSuccess)
                    throw ERROR("could not allocate managed memory");
                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) const {
                if (cudaFree(ptr) != cudaSuccess)
                    throw ERROR("could not free managed memory");
            }
        };
    } // namespace cuda
} // namespace backend