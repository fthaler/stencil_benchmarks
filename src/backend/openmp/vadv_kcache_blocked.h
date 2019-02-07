#pragma once

#include "backend/openmp/blocked_execution.h"
#include "backend/openmp/vadv_base.h"
#include "except.h"

namespace backend {
    namespace openmp {
        template <class Allocator>
        class vadv_kcache_blocked : public vadv_stencil_base<Allocator> {
          public:
            static void register_arguments(arguments &args) {
                vadv_stencil_base<Allocator>::register_arguments(args);
                blocked_execution_2d::register_arguments(args);
            }

            vadv_kcache_blocked(const arguments_map &args)
                : vadv_stencil_base<Allocator>(args), m_blocked_execution(args) {}

            void run() override {
                const real *__restrict__ ustage = this->m_ustage->data();
                const real *__restrict__ upos = this->m_upos->data();
                const real *__restrict__ utens = this->m_utens->data();
                real *__restrict__ utensstage = this->m_utensstage->data();
                const real *__restrict__ vstage = this->m_vstage->data();
                const real *__restrict__ vpos = this->m_vpos->data();
                const real *__restrict__ vtens = this->m_vtens->data();
                real *__restrict__ vtensstage = this->m_vtensstage->data();
                const real *__restrict__ wstage = this->m_wstage->data();
                const real *__restrict__ wpos = this->m_wpos->data();
                const real *__restrict__ wtens = this->m_wtens->data();
                real *__restrict__ wtensstage = this->m_wtensstage->data();
                real *__restrict__ ccol = this->m_ccol->data();
                real *__restrict__ dcol = this->m_dcol->data();
                const real *__restrict__ wcon = this->m_wcon->data();

                const int ksize = this->info().ksize();
                const int istride = this->info().istride();
                const int jstride = this->info().jstride();
                const int kstride = this->info().kstride();
                const auto block = m_blocked_execution.block(istride, jstride);

                const int u_inner_offset = block.inner.stride == istride ? 1 : 0;
                const int u_outer_offset = block.inner.stride == istride ? 0 : 1;
                const int v_inner_offset = block.inner.stride == istride ? 0 : 1;
                const int v_outer_offset = block.inner.stride == istride ? 1 : 0;

                if (block.inner.stride != 1)
                    throw ERROR("data must be contiguous along i- or j-axis");

#pragma omp parallel for collapse(2)
                for (int outer_ib = 0; outer_ib < block.outer.size; outer_ib += block.outer.blocksize) {
                    for (int inner_ib = 0; inner_ib < block.inner.size; inner_ib += block.inner.blocksize) {
                        const int inner_max = std::min(inner_ib + block.inner.blocksize, block.inner.size);
                        const int outer_max = std::min(outer_ib + block.outer.blocksize, block.outer.size);

                        for (int outer_i = outer_ib; outer_i < outer_max; ++outer_i) {
#pragma omp simd simdlen(64 / sizeof(real))
                            for (int inner_i = inner_ib; inner_i < inner_max; ++inner_i) {
                                this->forward_sweep(inner_i,
                                    outer_i,
                                    u_inner_offset,
                                    u_outer_offset,
                                    ccol,
                                    dcol,
                                    wcon,
                                    ustage,
                                    upos,
                                    utens,
                                    utensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);
                                this->backward_sweep(inner_i,
                                    outer_i,
                                    ccol,
                                    dcol,
                                    upos,
                                    utensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);

                                this->forward_sweep(inner_i,
                                    outer_i,
                                    v_inner_offset,
                                    v_outer_offset,
                                    ccol,
                                    dcol,
                                    wcon,
                                    vstage,
                                    vpos,
                                    vtens,
                                    vtensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);
                                this->backward_sweep(inner_i,
                                    outer_i,
                                    ccol,
                                    dcol,
                                    vpos,
                                    vtensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);

                                this->forward_sweep(inner_i,
                                    outer_i,
                                    0,
                                    0,
                                    ccol,
                                    dcol,
                                    wcon,
                                    wstage,
                                    wpos,
                                    wtens,
                                    wtensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);
                                this->backward_sweep(inner_i,
                                    outer_i,
                                    ccol,
                                    dcol,
                                    wpos,
                                    wtensstage,
                                    ksize,
                                    1,
                                    block.outer.stride,
                                    kstride);
                            }
                        }
                    }
                }
            }

          private:
            blocked_execution_2d m_blocked_execution;
        };
    } // namespace openmp
} // namespace backend
