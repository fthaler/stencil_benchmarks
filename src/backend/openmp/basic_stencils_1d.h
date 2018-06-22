#pragma once

#include "stencil/basic.h"
#include "stencil/basic_functors.h"

namespace backend {
    namespace openmp {
        namespace stencil {

            template <class Functor, class Allocator>
            class basic_base_1d : public ::stencil::basic<Functor, Allocator> {
              public:
                using ::stencil::basic<Functor, Allocator>::basic;

                void run() override {
                    const int ilast = this->info().last_index();
                    Functor functor(this->info(), this->m_src->data(), this->m_dst->data());

#pragma omp parallel for simd
#pragma vector nontemporal
                    for (int i = 0; i <= ilast; ++i)
                        functor(i);
                }
            };

            template <class Allocator>
            using basic_copy_1d = basic_base_1d<::stencil::copy_functor, Allocator>;

            template <class Allocator>
            using basic_avgi_1d = basic_base_1d<::stencil::avgi_functor, Allocator>;
            template <class Allocator>
            using basic_avgj_1d = basic_base_1d<::stencil::avgj_functor, Allocator>;
            template <class Allocator>
            using basic_avgk_1d = basic_base_1d<::stencil::avgk_functor, Allocator>;

            template <class Allocator>
            using basic_lapij_1d = basic_base_1d<::stencil::lapij_functor, Allocator>;
            template <class Allocator>
            using basic_lapik_1d = basic_base_1d<::stencil::lapik_functor, Allocator>;
            template <class Allocator>
            using basic_lapjk_1d = basic_base_1d<::stencil::lapjk_functor, Allocator>;
            template <class Allocator>
            using basic_lapijk_1d = basic_base_1d<::stencil::lapijk_functor, Allocator>;

        } // namespace stencil
    }     // namespace openmp
} // namespace backend
