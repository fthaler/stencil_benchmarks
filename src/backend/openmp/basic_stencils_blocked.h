#pragma once

#include "backend/openmp/blocked_execution.h"
#include "stencil/basic.h"
#include "stencil/basic_functors.h"

namespace backend {
    namespace openmp {

        template <class Functor, class Allocator>
        class basic_base_blocked : public stencil::basic<Functor, Allocator> {
          public:
            static void register_arguments(arguments &args) {
                stencil::basic<Functor, Allocator>::register_arguments(args);
                blocked_execution::register_arguments(args);
            }

            basic_base_blocked(const arguments_map &args)
                : stencil::basic<Functor, Allocator>(args), m_blocked_execution(args) {}

            void run() override {
                const Functor functor(this->info(), this->m_src->data(), this->m_dst->data());
                const auto block =
                    m_blocked_execution.block(this->info().istride(), this->info().jstride(), this->info().kstride());

                if (block.inner.stride != 1)
                    throw ERROR("data must be contiguous along one axis");

#pragma omp parallel for collapse(3)
                for (int outer_ib = 0; outer_ib < block.outer.size; outer_ib += block.outer.blocksize) {
                    for (int middle_ib = 0; middle_ib < block.middle.size; middle_ib += block.middle.blocksize) {
                        for (int inner_ib = 0; inner_ib < block.inner.size; inner_ib += block.inner.blocksize) {
                            const int outer_imax = outer_ib + block.outer.blocksize <= block.outer.size
                                                       ? outer_ib + block.outer.blocksize
                                                       : block.outer.size;
                            const int middle_imax = middle_ib + block.middle.blocksize <= block.middle.size
                                                        ? middle_ib + block.middle.blocksize
                                                        : block.middle.size;
                            const int inner_imax = inner_ib + block.inner.blocksize <= block.inner.size
                                                       ? inner_ib + block.inner.blocksize
                                                       : block.inner.size;

                            int index = outer_ib * block.outer.stride + middle_ib * block.middle.stride +
                                        inner_ib * block.inner.stride;

                            for (int outer_i = outer_ib; outer_i < outer_imax; ++outer_i) {
                                for (int middle_i = middle_ib; middle_i < middle_imax; ++middle_i) {
#pragma omp simd
#pragma vector nontemporal
                                    for (int inner_i = inner_ib; inner_i < inner_imax; ++inner_i) {
                                        functor(index);
                                        ++index;
                                    }

                                    index += block.middle.stride - (inner_imax - inner_ib);
                                }
                                index += block.outer.stride - (middle_imax - middle_ib) * block.middle.stride;
                            }
                        }
                    }
                }
            }

          private:
            blocked_execution m_blocked_execution;
        };

        template <class Allocator>
        using basic_copy_blocked = basic_base_blocked<stencil::copy_functor, Allocator>;

        template <class Allocator>
        using basic_avgi_blocked = basic_base_blocked<stencil::avgi_functor, Allocator>;
        template <class Allocator>
        using basic_avgj_blocked = basic_base_blocked<stencil::avgj_functor, Allocator>;
        template <class Allocator>
        using basic_avgk_blocked = basic_base_blocked<stencil::avgk_functor, Allocator>;

        template <class Allocator>
        using basic_lapij_blocked = basic_base_blocked<stencil::lapij_functor, Allocator>;
        template <class Allocator>
        using basic_lapik_blocked = basic_base_blocked<stencil::lapik_functor, Allocator>;
        template <class Allocator>
        using basic_lapjk_blocked = basic_base_blocked<stencil::lapjk_functor, Allocator>;
        template <class Allocator>
        using basic_lapijk_blocked = basic_base_blocked<stencil::lapijk_functor, Allocator>;

    } // namespace openmp
} // namespace backend
