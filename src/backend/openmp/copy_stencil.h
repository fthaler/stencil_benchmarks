#pragma once

#include "stencils/copy_stencil.h"

namespace backend {
    namespace openmp {

        template <class T, class Allocator, class Functor>
        class copy_stencil_1d : public copy_stencil<T, Allocator> {
          public:
            void run() override {
                const T *__restrict__ src = this->m_src.data();
                T *__restrict__ dst = this->m_dst.data();
                const int ilast = this->info().last_index();

                Functor functor(src, dst);

#pragma omp parallel for simd
                for (int i = 0; i <= ilast; ++i)
                    functor(i);
            }
        };

    } // namespace openmp
} // namespace backend
