#pragma once

#include <cstdlib>

#include "except.h"

namespace platform {
    namespace knl {

        template <class ValueType>
        struct allocator {
            using value_type = ValueType;
            static constexpr std::size_t alignment = 2048 * 1024;

            template <class OtherValueType>
            struct rebind {
                using other = allocator<OtherValueType>;
            };

            value_type *allocate(std::size_t n) const {
                static std::size_t offset = 64;
                char *raw_ptr;
                if (posix_memalign(reinterpret_cast<void **>(&raw_ptr), alignment, n * sizeof(value_type) + offset))
                    throw ERROR("could not allocate memory");

                value_type *ptr = reinterpret_cast<value_type *>(raw_ptr + offset);

                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                *offset_ptr = offset;
                if ((offset *= 2) >= 16384)
                    offset = 64;

                return ptr;
            }

            void deallocate(value_type *ptr, std::size_t) const {
                std::size_t *offset_ptr = reinterpret_cast<std::size_t *>(ptr) - 1;
                std::size_t offset = *offset_ptr;

                char *raw_ptr = reinterpret_cast<char *>(ptr) - offset;
                free(raw_ptr);
            }

            template <class OtherValueType, class... Args>
            void construct(OtherValueType *, Args &&...) {}
        };

    } // namespace knl
} // namespace platform
