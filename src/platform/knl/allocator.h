#pragma once

#include <cstdlib>

#include <sys/mman.h>

#include "except.h"

namespace platform {
    namespace knl {

        template <class ValueType>
        struct allocator {
            using value_type = ValueType;

            template <class OtherValueType>
            struct rebind {
                using other = allocator<OtherValueType>;
            };

            value_type *allocate(std::size_t n) const {
                static std::size_t offset = 64;
                char *raw_ptr = reinterpret_cast<char *>(mmap(nullptr,
                    n * sizeof(value_type) + offset,
                    PROT_READ | PROT_WRITE,
                    MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE | MAP_HUGETLB,
                    -1,
                    0));
                if (raw_ptr == reinterpret_cast<char *>(-1))
                    throw ERROR("could not allocate memory");

                value_type *ptr = reinterpret_cast<value_type *>(raw_ptr + offset);

                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                *offset_ptr = offset;
                if ((offset *= 2) >= 16384)
                    offset = 64;

                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t n) const {
                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                std::size_t offset = *offset_ptr;

                char *raw_ptr = reinterpret_cast<char *>(ptr) - offset;
                munmap(raw_ptr, n * sizeof(value_type) + offset);
            }

            template <class OtherValueType, class... Args>
            void construct(OtherValueType *, Args &&...) {}
        };

    } // namespace knl
} // namespace platform
