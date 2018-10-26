#pragma once

#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace openmp {

        template <class Allocator>
        class hdiff_naive : public stencil::hdiff<Allocator> {
          public:
            static void register_arguments(arguments &args) { stencil::hdiff<Allocator>::register_arguments(args); }

            hdiff_naive(const arguments_map &args)
                : stencil::hdiff<Allocator>(args), m_lap(this->template create_field<real, Allocator>()),
                  m_flx(this->template create_field<real, Allocator>()),
                  m_fly(this->template create_field<real, Allocator>()) {}

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
                real *__restrict__ lap = m_lap->data();
                real *__restrict__ flx = m_flx->data();
                real *__restrict__ fly = m_fly->data();
                real *__restrict__ dst = this->m_dst->data();

#pragma omp parallel
                {
#if defined(__GNUC__) && __GNUC__ < 7
#pragma omp for collapse(3)
#else
#pragma omp for simd collapse(3)
#endif
                    for (int k = 0; k < ksize; ++k) {
                        for (int j = -2; j < jsize + 2; ++j) {
                            for (int i = -2; i < isize + 2; ++i) {
                                const int index = i * istride + j * jstride + k * kstride;

                                lap[index] = 4 * src[index] - (src[index - istride] + src[index + istride] +
                                                                  src[index - jstride] + src[index + jstride]);
                            }
                        }
                    }

#if defined(__GNUC__) && __GNUC__ < 7
#pragma omp for collapse(3)
#else
#pragma omp for simd collapse(3)
#endif
                    for (int k = 0; k < ksize; ++k) {
                        for (int j = 0; j < jsize; ++j) {
                            for (int i = -1; i < isize + 1; ++i) {
                                const int index = i * istride + j * jstride + k * kstride;
                                flx[index] = lap[index + istride] - lap[index];
                                if (flx[index] * (src[index + istride] - src[index]) > 0)
                                    flx[index] = 0;
                            }
                        }
                    }

#if defined(__GNUC__) && __GNUC__ < 7
#pragma omp for collapse(3)
#else
#pragma omp for simd collapse(3)
#endif
                    for (int k = 0; k < ksize; ++k) {
                        for (int j = -1; j < jsize + 1; ++j) {
                            for (int i = 0; i < isize; ++i) {
                                const int index = i * istride + j * jstride + k * kstride;
                                fly[index] = lap[index + jstride] - lap[index];
                                if (fly[index] * (src[index + jstride] - src[index]) > 0)
                                    fly[index] = 0;
                            }
                        }
                    }

#if defined(__GNUC__) && __GNUC__ < 7
#pragma omp for collapse(3)
#else
#pragma omp for simd collapse(3)
#endif
                    for (int k = 0; k < ksize; ++k) {
                        for (int j = 0; j < jsize; ++j) {
                            for (int i = 0; i < isize; ++i) {
                                const int index = i * istride + j * jstride + k * kstride;
                                dst[index] = src[index] - coeff[index] * (flx[index] - flx[index - istride] +
                                                                             fly[index] - fly[index - jstride]);
                            }
                        }
                    }
                }
            }

          private:
            field_ptr<real, Allocator> m_lap, m_flx, m_fly;
        };
    } // namespace openmp
} // namespace backend
