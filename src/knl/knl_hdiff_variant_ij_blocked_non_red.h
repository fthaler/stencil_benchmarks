#pragma once

#include "knl/knl_hdiff_stencil_variant.h"

namespace platform {

    namespace knl {

        template <class Platform, class ValueType>
        class knl_hdiff_variant_ij_blocked_non_red final : public knl_hdiff_stencil_variant<Platform, ValueType> {
          public:
            using value_type = ValueType;

            knl_hdiff_variant_ij_blocked_non_red(const arguments_map &args)
                : knl_hdiff_stencil_variant<Platform, ValueType>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0 || m_jblocksize <= 0)
                    throw ERROR("invalid block size");
            }

            void hdiff() override {

                const value_type *__restrict__ in = this->in();
                const value_type *__restrict__ coeff = this->coeff();
                value_type *__restrict__ lap = this->lap();
                value_type *__restrict__ flx = this->flx();
                value_type *__restrict__ fly = this->fly();
                value_type *__restrict__ out = this->out();  

                constexpr int istride = 1;
                const int jstride = this->jstride();
                const int kstride = this->kstride();
                const int h = this->halo();
                const int isize = this->isize();
                const int jsize = this->jsize();
                const int ksize = this->ksize();

                if (this->istride() != 1)
                    throw ERROR("this variant is only compatible with unit i-stride layout");

                for (int k = 0; k < ksize; ++k) {
                    #pragma omp parallel for collapse(2) 
                    for (int jb = 0; jb < jsize+2; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize+2; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize+2 ? ib + m_iblocksize : isize+2;
                            const int jmax = jb + m_jblocksize <= jsize+2 ? jb + m_jblocksize : jsize+2;
                            int index = (ib-1) * istride + (jb-1) * jstride + k * kstride;
                            for (int j = jb; j < jmax; ++j) {
                                #pragma omp simd
                                #pragma vector nontemporal
                                for (int i = ib; i < imax; ++i) {
                                    lap[index] = 4 * in[index] -
                                        (in[index - istride] + in[index + istride] + in[index - jstride] + in[index + jstride]);
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                        }
                    }

                    #pragma omp parallel for collapse(2) 
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize+1; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize+1 ? ib + m_iblocksize : isize+1;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;
                            int index = (ib-1) * istride + jb * jstride + k * kstride;
                            for (int j = jb; j < jmax; ++j) {
                                #pragma omp simd
                                #pragma vector nontemporal
                                for (int i = ib; i < imax; ++i) {
                                    flx[index] = lap[index + istride] - lap[index];
                                    if (flx[index] * (in[index + istride] - in[index]) > 0)
                                        flx[index] = 0.;
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                        }
                    }

                    #pragma omp parallel for collapse(2) 
                    for (int jb = 0; jb < jsize+1; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize+1 ? jb + m_jblocksize : jsize+1;
                            int index = ib * istride + (jb-1) * jstride + k * kstride;
                            for (int j = jb; j < jmax; ++j) {
                                #pragma omp simd
                                #pragma vector nontemporal
                                for (int i = ib; i < imax; ++i) {
                                    fly[index] = lap[index + jstride] - lap[index];
                                    if (fly[index] * (in[index + jstride] - in[index]) > 0)
                                        fly[index] = 0.;
                                    index += istride;
                                }
                                index += jstride - (imax - ib) * istride;
                            }
                        }
                    }

                    #pragma omp parallel for collapse(2) 
                    for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                        for (int ib = 0; ib < isize; ib += m_iblocksize) {
                            const int imax = ib + m_iblocksize <= isize ? ib + m_iblocksize : isize;
                            const int jmax = jb + m_jblocksize <= jsize ? jb + m_jblocksize : jsize;

                            int index = ib * istride + jb * jstride + k * kstride;
                            for (int j = jb; j < jmax; ++j) {
                                #pragma omp simd
                                #pragma vector nontemporal
                                for (int i = ib; i < imax; ++i) {
                                    out[index] =
                                        in[index] - coeff[index] *
                                            (flx[index] - flx[index - istride] + fly[index] - fly[index - jstride]);
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

    } // namespace knl

} // namespace platform
