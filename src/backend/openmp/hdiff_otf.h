#pragma once

#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace openmp {

        template <class Allocator>
        class hdiff_otf : public stencil::hdiff<Allocator> {
          public:
            using stencil::hdiff<Allocator>::hdiff;

            static void register_arguments(arguments &args) { stencil::hdiff<Allocator>::register_arguments(args); }

            void run() override {
                const int isize = this->info().isize();
                const int jsize = this->info().jsize();
                const int ksize = this->info().ksize();
                if (this->info().istride() != 1)
                    throw ERROR("i-stride must be 1");
                constexpr int istride = 1;
                const int jstride = this->info().jstride();
                const int kstride = this->info().kstride();

                const real *__restrict__ src = this->m_src->data();
                const real *__restrict__ coeff = this->m_coeff->data();
                real *__restrict__ dst = this->m_dst->data();

#if defined(__GNUC__) && __GNUC__ < 7
#pragma omp parallel for collapse(3)
#else
#pragma omp parallel for simd collapse(3)
#endif
                for (int k = 0; k < ksize; ++k) {
                    for (int j = 0; j < jsize; ++j) {
                        for (int i = 0; i < isize; ++i) {
                            const int index = i * istride + j * jstride + k * kstride;
                            real lap_ij = 4 * src[index] - src[index - istride] - src[index + istride] -
                                          src[index - jstride] - src[index + jstride];
                            real lap_imj = 4 * src[index - istride] - src[index - 2 * istride] - src[index] -
                                           src[index - istride - jstride] - src[index - istride + jstride];
                            real lap_ipj = 4 * src[index + istride] - src[index] - src[index + 2 * istride] -
                                           src[index + istride - jstride] - src[index + istride + jstride];
                            real lap_ijm = 4 * src[index - jstride] - src[index - istride - jstride] -
                                           src[index + istride - jstride] - src[index - 2 * jstride] - src[index];
                            real lap_ijp = 4 * src[index + jstride] - src[index - istride + jstride] -
                                           src[index + istride + jstride] - src[index] - src[index + 2 * jstride];

                            real flx_ij = lap_ipj - lap_ij;
                            flx_ij = flx_ij * (src[index + istride] - src[index]) > 0 ? 0 : flx_ij;

                            real flx_imj = lap_ij - lap_imj;
                            flx_imj = flx_imj * (src[index] - src[index - istride]) > 0 ? 0 : flx_imj;

                            real fly_ij = lap_ijp - lap_ij;
                            fly_ij = fly_ij * (src[index + jstride] - src[index]) > 0 ? 0 : fly_ij;

                            real fly_ijm = lap_ij - lap_ijm;
                            fly_ijm = fly_ijm * (src[index] - src[index - jstride]) > 0 ? 0 : fly_ijm;

                            dst[index] = src[index] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
                        }
                    }
                }
            }
        };
    } // namespace openmp
} // namespace backend
