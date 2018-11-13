#pragma once

#include "backend/openmp/blocked_execution.h"
#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace openmp {

        template <class Allocator>
        class hdiff_otf_blocked : public stencil::hdiff<Allocator> {
          public:
            using stencil::hdiff<Allocator>::hdiff;

            static void register_arguments(arguments &args) {
                stencil::hdiff<Allocator>::register_arguments(args);
                blocked_execution_2d::register_arguments(args);
            }

            hdiff_otf_blocked(const arguments_map &args) : stencil::hdiff<Allocator>(args), m_blocked_execution(args) {}

            void run() override {
                const real *__restrict__ src = this->m_src->data();
                const real *__restrict__ coeff = this->m_coeff->data();
                real *__restrict__ dst = this->m_dst->data();

                const int ksize = this->info().ksize();
                const int istride = this->info().istride();
                const int jstride = this->info().jstride();
                const int kstride = this->info().kstride();
                auto block = m_blocked_execution.block(istride, jstride);

                if (block.inner.stride != 1)
                    throw ERROR("data must be contiguous along i- or j-axis");

#pragma omp parallel for collapse(3)
                for (int k = 0; k < ksize; ++k) {
                    for (int outer_ib = 0; outer_ib < block.outer.size; outer_ib += block.outer.blocksize) {
                        for (int inner_ib = 0; inner_ib < block.inner.size; inner_ib += block.inner.blocksize) {
                            int index = inner_ib * block.inner.stride + outer_ib * block.outer.stride + k * kstride;
                            const int inner_max = std::min(inner_ib + block.inner.blocksize, block.inner.size);
                            const int outer_max = std::min(outer_ib + block.outer.blocksize, block.outer.size);

                            for (int outer_i = outer_ib; outer_i < outer_max; ++outer_i) {
#pragma omp simd
#pragma vector nontemporal
                                for (int inner_i = inner_ib; inner_i < inner_max; ++inner_i) {
                                    real lap_ij = 4 * src[index] - src[index - istride] - src[index + istride] -
                                                  src[index - jstride] - src[index + jstride];
                                    real lap_imj = 4 * src[index - istride] - src[index - 2 * istride] - src[index] -
                                                   src[index - istride - jstride] - src[index - istride + jstride];
                                    real lap_ipj = 4 * src[index + istride] - src[index] - src[index + 2 * istride] -
                                                   src[index + istride - jstride] - src[index + istride + jstride];
                                    real lap_ijm = 4 * src[index - jstride] - src[index - istride - jstride] -
                                                   src[index + istride - jstride] - src[index - 2 * jstride] -
                                                   src[index];
                                    real lap_ijp = 4 * src[index + jstride] - src[index - istride + jstride] -
                                                   src[index + istride + jstride] - src[index] -
                                                   src[index + 2 * jstride];

                                    real flx_ij = lap_ipj - lap_ij;
                                    flx_ij = flx_ij * (src[index + istride] - src[index]) > 0 ? 0 : flx_ij;

                                    real flx_imj = lap_ij - lap_imj;
                                    flx_imj = flx_imj * (src[index] - src[index - istride]) > 0 ? 0 : flx_imj;

                                    real fly_ij = lap_ijp - lap_ij;
                                    fly_ij = fly_ij * (src[index + jstride] - src[index]) > 0 ? 0 : fly_ij;

                                    real fly_ijm = lap_ij - lap_ijm;
                                    fly_ijm = fly_ijm * (src[index] - src[index - jstride]) > 0 ? 0 : fly_ijm;

                                    dst[index] = src[index] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
                                    ++index;
                                }
                                index += block.outer.stride - (inner_max - inner_ib);
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
