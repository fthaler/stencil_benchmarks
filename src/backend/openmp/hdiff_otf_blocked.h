#pragma once

#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace openmp {
        namespace stencil {

            template <class Allocator>
            class hdiff_otf_blocked : public ::stencil::hdiff<Allocator> {
              public:
                using ::stencil::hdiff<Allocator>::hdiff;

                static void register_arguments(arguments &args) {
                    ::stencil::hdiff<Allocator>::register_arguments(args);
                    args.add({"i-blocksize", "block size in i-direction", "32"})
                        .add({"j-blocksize", "block size in j-direction", "32"});
                }

                hdiff_otf_blocked(const arguments_map &args)
                    : ::stencil::hdiff<Allocator>(args), m_iblocksize(args.get<int>("i-blocksize")),
                      m_jblocksize(args.get<int>("j-blocksize")) {
                    if (m_iblocksize <= 0)
                        throw ERROR("invalid i-blocksize");
                    if (m_jblocksize <= 0)
                        throw ERROR("invalid j-blocksize");
                }

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

#pragma omp parallel for collapse(3)
                    for (int k = 0; k < ksize; ++k) {
                        for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                            for (int ib = 0; ib < isize; ib += m_iblocksize) {
                                int index = ib * istride + jb * jstride + k * kstride;
                                const int imax = std::min(ib + m_iblocksize, isize);
                                const int jmax = std::min(jb + m_jblocksize, jsize);

                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
#pragma vector nontemporal
                                    for (int i = ib; i < imax; ++i) {
                                        real lap_ij = 4 * src[index] - src[index - istride] - src[index + istride] -
                                                      src[index - jstride] - src[index + jstride];
                                        real lap_imj = 4 * src[index - istride] - src[index - 2 * istride] -
                                                       src[index] - src[index - istride - jstride] -
                                                       src[index - istride + jstride];
                                        real lap_ipj = 4 * src[index + istride] - src[index] -
                                                       src[index + 2 * istride] - src[index + istride - jstride] -
                                                       src[index + istride + jstride];
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
                                        index += istride;
                                    }
                                    index += jstride - (imax - ib) * istride;
                                }
                            }
                        }
                    }
                }

              private:
                int m_iblocksize, m_jblocksize;
            };
        } // namespace stencil
    }     // namespace openmp
} // namespace backend
