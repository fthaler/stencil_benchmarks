#pragma once

#include "stencil/vadv.h"

#ifdef __SSE__
#include <xmmintrin.h>
#endif

namespace backend {
    namespace openmp {

        template <class Allocator>
        class vadv_stencil_base : public ::stencil::vadv<Allocator> {
          public:
            using ::stencil::vadv<Allocator>::vadv;

          protected:
#pragma omp declare simd linear(i) uniform(j, ccol, dcol, datacol, upos, utensstage, ksize, istride, jstride, kstride)
            void backward_sweep_kmax(const int i,
                const int j,
                const real *__restrict__ ccol,
                const real *__restrict__ dcol,
                real *__restrict__ datacol,
                const real *__restrict__ upos,
                real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                const int datacol_index = i * istride + j * jstride;
                datacol[datacol_index] = dcol[index];
                utensstage[index] = this->dtr_stage * (datacol[datacol_index] - upos[index]);
            }

#pragma omp declare simd linear(i) \
    uniform(j, k, ccol, dcol, datacol, upos, utensstage, ksize, istride, jstride, kstride)
            void backward_sweep_kbody(const int i,
                const int j,
                const int k,
                const real *__restrict__ ccol,
                const real *__restrict__ dcol,
                real *__restrict__ datacol,
                const real *__restrict__ upos,
                real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int index = i * istride + j * jstride + k * kstride;
                const int datacol_index = i * istride + j * jstride;
                datacol[datacol_index] = dcol[index] - ccol[index] * datacol[datacol_index];
                utensstage[index] = this->dtr_stage * (datacol[datacol_index] - upos[index]);
            }
#pragma omp declare simd linear(i) uniform(j, ccol, dcol, upos, utensstage, ksize, istride, jstride, kstride)
            void backward_sweep(const int i,
                const int j,
                const real *__restrict__ ccol,
                const real *__restrict__ dcol,
                const real *__restrict__ upos,
                real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr real dtr_stage = 3.0 / 20.0;

                real datacol;

                int index = i * istride + j * jstride + (ksize - 1) * kstride;
                // k
                {
                    datacol = dcol[index];
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
#ifdef __SSE__
                    /*constexpr int prefdist = 6;
                    if (k >= prefdist) {
                        const int prefindex = index - prefdist * kstride;
                        _mm_prefetch(reinterpret_cast<const char *>(&dcol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ccol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&upos[prefindex]), _MM_HINT_NTA);
                        _mm_prefetch(reinterpret_cast<const char *>(&utensstage[prefindex]), _MM_HINT_NTA);
                    }*/
#endif
                    datacol = dcol[index] - ccol[index] * datacol;
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }
            }
#pragma omp declare simd linear(i) \
    uniform(j, k, ccol, dcol, datacol, upos, utensstage, ksize, istride, jstride, kstride)
            void backward_sweep_k(const int i,
                const int j,
                const int k,
                const real *__restrict__ ccol,
                const real *__restrict__ dcol,
                real *__restrict__ datacol,
                const real *__restrict__ upos,
                real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                constexpr real dtr_stage = 3.0 / 20.0;

                if (k == ksize - 1) {
                    backward_sweep_kmax(i, j, ccol, dcol, datacol, upos, utensstage, ksize, istride, jstride, kstride);
                } else {
                    backward_sweep_kbody(
                        i, j, k, ccol, dcol, datacol, upos, utensstage, ksize, istride, jstride, kstride);
                }
            }

#pragma omp declare simd linear(i) \
    uniform(j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, ksize, istride, jstride, kstride)
            void forward_sweep_kmin(const int i,
                const int j,
                const int ishift,
                const int jshift,
                real *__restrict__ ccol,
                real *__restrict__ dcol,
                const real *__restrict__ wcon,
                const real *__restrict__ ustage,
                const real *__restrict__ upos,
                const real *__restrict__ utens,
                const real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int k = 0;
                const int index = i * istride + j * jstride + k * kstride;
                real gcv =
                    real(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);
                real cs = gcv * this->bet_m;

                ccol[index] = gcv * this->bet_p;
                real bcol = this->dtr_stage - ccol[index];

                real correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / bcol;
                ccol[index] = ccol[index] * divided;
                dcol[index] = dcol[index] * divided;
            }
#pragma omp declare simd linear(i) \
    uniform(j, k, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, ksize, istride, jstride, kstride)
            void forward_sweep_kbody(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                real *__restrict__ ccol,
                real *__restrict__ dcol,
                const real *__restrict__ wcon,
                const real *__restrict__ ustage,
                const real *__restrict__ upos,
                const real *__restrict__ utens,
                const real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int index = i * istride + j * jstride + k * kstride;
                real gav = real(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                real gcv =
                    real(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                real as = gav * this->bet_m;
                real cs = gcv * this->bet_m;

                real acol = gav * this->bet_p;
                ccol[index] = gcv * this->bet_p;
                real bcol = this->dtr_stage - acol - ccol[index];

                real correction_term =
                    -as * (ustage[index - kstride] - ustage[index]) - cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / (bcol - ccol[index - kstride] * acol);
                ccol[index] = ccol[index] * divided;
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }

#pragma omp declare simd linear(i) \
    uniform(j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, ksize, istride, jstride, kstride)
            void forward_sweep_kmax(const int i,
                const int j,
                const int ishift,
                const int jshift,
                real *__restrict__ ccol,
                real *__restrict__ dcol,
                const real *__restrict__ wcon,
                const real *__restrict__ ustage,
                const real *__restrict__ upos,
                const real *__restrict__ utens,
                const real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                real gav = real(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                real as = gav * this->bet_m;

                real acol = gav * this->bet_p;
                real bcol = this->dtr_stage - acol;

                real correction_term = -as * (ustage[index - kstride] - ustage[index]);
                dcol[index] = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / (bcol - ccol[index - kstride] * acol);
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }
#pragma omp declare simd linear(i) \
    uniform(j, k, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, ksize, istride, jstride, kstride)
            void forward_sweep_k(const int i,
                const int j,
                const int k,
                const int ishift,
                const int jshift,
                real *__restrict__ ccol,
                real *__restrict__ dcol,
                const real *__restrict__ wcon,
                const real *__restrict__ ustage,
                const real *__restrict__ upos,
                const real *__restrict__ utens,
                const real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                if (k == 0) {
                    forward_sweep_kmin(i,
                        j,
                        ishift,
                        jshift,
                        ccol,
                        dcol,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        ksize,
                        istride,
                        jstride,
                        kstride);
                } else if (k == ksize - 1) {
                    forward_sweep_kmax(i,
                        j,
                        ishift,
                        jshift,
                        ccol,
                        dcol,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        ksize,
                        istride,
                        jstride,
                        kstride);
                } else {
                    forward_sweep_kbody(i,
                        j,
                        k,
                        ishift,
                        jshift,
                        ccol,
                        dcol,
                        wcon,
                        ustage,
                        upos,
                        utens,
                        utensstage,
                        ksize,
                        istride,
                        jstride,
                        kstride);
                }
            }
#pragma omp declare simd linear(i) \
    uniform(j, ishift, jshift, ccol, wcon, ustage, upos, utens, utensstage, ksize, istride, jstride, kstride)
            void forward_sweep(const int i,
                const int j,
                const int ishift,
                const int jshift,
                real *__restrict__ ccol,
                real *__restrict__ dcol,
                const real *__restrict__ wcon,
                const real *__restrict__ ustage,
                const real *__restrict__ upos,
                const real *__restrict__ utens,
                const real *__restrict__ utensstage,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {

                real ccol0, ccol1;
                real dcol0, dcol1;
                real ustage0, ustage1, ustage2;
                real wcon0, wcon1;
                real wcon_shift0, wcon_shift1;

                int index = i * istride + j * jstride;
                // k minimum
                {
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    real gcv = real(0.25) * (wcon_shift0 + wcon0);
                    real cs = gcv * this->bet_m;

                    ccol0 = gcv * this->bet_p;
                    real bcol = this->dtr_stage - ccol0;

                    ustage0 = ustage[index + kstride];
                    ustage1 = ustage[index];
                    real correction_term = -cs * (ustage0 - ustage1);
                    dcol0 = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    real divided = real(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }
                // k body
                for (int k = 1; k < ksize - 1; ++k) {
#ifdef __SSE__
                    constexpr int prefdist = 3;
                    if (k < ksize - prefdist) {
                        const int prefindex = index + prefdist * kstride;
                        _mm_prefetch(reinterpret_cast<const char *>(&upos[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ustage[prefindex + kstride]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&utens[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&utensstage[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&wcon[prefindex + kstride]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(
                                         &wcon[prefindex + ishift * istride + jshift * jstride + kstride]),
                            _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&ccol[prefindex]), _MM_HINT_T1);
                        _mm_prefetch(reinterpret_cast<const char *>(&dcol[prefindex]), _MM_HINT_T1);
                    }
#else
#warn "no sofware prefetching in vadv stencil"
#endif

                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    real gav = real(-0.25) * (wcon_shift1 + wcon1);
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    real gcv = real(0.25) * (wcon_shift0 + wcon0);

                    real as = gav * this->bet_m;
                    real cs = gcv * this->bet_m;

                    real acol = gav * this->bet_p;
                    ccol0 = gcv * this->bet_p;
                    real bcol = this->dtr_stage - acol - ccol0;

                    ustage0 = ustage[index + kstride];
                    real correction_term = -as * (ustage2 - ustage1) - cs * (ustage0 - ustage1);
                    dcol0 = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    real divided = real(1.0) / (bcol - ccol1 * acol);
                    ccol0 = ccol0 * divided;
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }
                // k maximum
                {
                    ccol1 = ccol0;
                    dcol1 = dcol0;
                    ustage2 = ustage1;
                    ustage1 = ustage0;
                    wcon1 = wcon0;
                    wcon_shift1 = wcon_shift0;

                    real gav = real(-0.25) * (wcon_shift1 + wcon1);

                    real as = gav * this->bet_m;

                    real acol = gav * this->bet_p;
                    real bcol = this->dtr_stage - acol;

                    real correction_term = -as * (ustage2 - ustage1);
                    dcol0 = this->dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    real divided = real(1.0) / (bcol - ccol1 * acol);
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }
            }
        };

    } // namespace openmp
} // namespace backend
