#include "backend/cuda/block_index.h"
#include "backend/cuda/check.h"
#include "backend/cuda/vadv_kcache.h"
#include "except.h"

namespace backend {
    namespace cuda {
        __device__ __forceinline__ void backward_sweep(const real *ccol,
            const real *__restrict__ dcol,
            const real *__restrict__ upos,
            real *__restrict__ utensstage,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            using block_index_t = vadv_kcache::blocked_execution_t::block_index_t;

            const block_index_t bidx(isize, jsize);
            int index = bidx.i * istride + bidx.j * jstride + (ksize - 1) * kstride;

            constexpr real dtr_stage = 3.0 / 20.0;

            real datacol;
            if (bidx.in_block()) {
                // k maximum
                {
                    datacol = dcol[index];
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    datacol = dcol[index] - ccol[index] * datacol;
                    utensstage[index] = dtr_stage * (datacol - upos[index]);

                    index -= kstride;
                }
            }
        }

        __device__ __forceinline__ void forward_sweep(const int ishift,
            const int jshift,
            real *__restrict__ ccol,
            real *__restrict__ dcol,
            const real *__restrict__ wcon,
            const real *__restrict__ ustage,
            const real *__restrict__ upos,
            const real *__restrict__ utens,
            const real *__restrict__ utensstage,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            using block_index_t = vadv_kcache::blocked_execution_t::block_index_t;

            const block_index_t bidx(isize, jsize);
            int index = bidx.i * istride + bidx.j * jstride;

            constexpr real dtr_stage = 3.0 / 20.0;
            constexpr real beta_v = 0;
            constexpr real bet_m = 0.5 * (1.0 - beta_v);
            constexpr real bet_p = 0.5 * (1.0 + beta_v);

            real ccol0, ccol1;
            real dcol0, dcol1;
            real ustage0, ustage1, ustage2;
            real wcon0, wcon1;
            real wcon_shift0, wcon_shift1;

            if (bidx.in_block()) {
                // k minimum
                {
                    wcon_shift0 = wcon[index + ishift * istride + jshift * jstride + kstride];
                    wcon0 = wcon[index + kstride];
                    real gcv = real(0.25) * (wcon_shift0 + wcon0);
                    real cs = gcv * bet_m;

                    ccol0 = gcv * bet_p;
                    real bcol = dtr_stage - ccol0;

                    ustage0 = ustage[index + kstride];
                    ustage1 = ustage[index];
                    real correction_term = -cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    real divided = real(1.0) / bcol;
                    ccol0 = ccol0 * divided;
                    dcol0 = dcol0 * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;

                    index += kstride;
                }

                // k body
                for (int k = 1; k < ksize - 1; ++k) {
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

                    real as = gav * bet_m;
                    real cs = gcv * bet_m;

                    real acol = gav * bet_p;
                    ccol0 = gcv * bet_p;
                    real bcol = dtr_stage - acol - ccol0;

                    ustage0 = ustage[index + kstride];
                    real correction_term = -as * (ustage2 - ustage1) - cs * (ustage0 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

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

                    real as = gav * bet_m;

                    real acol = gav * bet_p;
                    real bcol = dtr_stage - acol;

                    real correction_term = -as * (ustage2 - ustage1);
                    dcol0 = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                    real divided = real(1.0) / (bcol - ccol1 * acol);
                    dcol0 = (dcol0 - dcol1 * acol) * divided;

                    ccol[index] = ccol0;
                    dcol[index] = dcol0;
                }
            }
        }

        __global__ void vadv_kcache_kernel(const real *ustage,
            const real *__restrict__ upos,
            const real *__restrict__ utens,
            real *__restrict__ utensstage,
            const real *__restrict__ vstage,
            const real *__restrict__ vpos,
            const real *__restrict__ vtens,
            real *__restrict__ vtensstage,
            const real *__restrict__ wstage,
            const real *__restrict__ wpos,
            const real *__restrict__ wtens,
            real *__restrict__ wtensstage,
            real *__restrict__ ccol,
            real *__restrict__ dcol,
            const real *__restrict__ wcon,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            forward_sweep(1,
                0,
                ccol,
                dcol,
                wcon,
                ustage,
                upos,
                utens,
                utensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, upos, utensstage, isize, jsize, ksize, istride, jstride, kstride);

            forward_sweep(0,
                1,
                ccol,
                dcol,
                wcon,
                vstage,
                vpos,
                vtens,
                vtensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, vpos, vtensstage, isize, jsize, ksize, istride, jstride, kstride);

            forward_sweep(0,
                0,
                ccol,
                dcol,
                wcon,
                wstage,
                wpos,
                wtens,
                wtensstage,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);
            backward_sweep(ccol, dcol, wpos, wtensstage, isize, jsize, ksize, istride, jstride, kstride);
        }

        void vadv_kcache::register_arguments(arguments &args) {
            stencil::vadv<allocator<real>>::register_arguments(args);
            blocked_execution_t::register_arguments(args);
        }

        vadv_kcache::vadv_kcache(const arguments_map &args)
            : stencil::vadv<allocator<real>>(args), m_blocked_execution(args) {
            CUDA_CHECK(cudaFuncSetCacheConfig(vadv_kcache_kernel, cudaFuncCachePreferL1));
        }

        void vadv_kcache::run() {
            const int isize = this->info().isize();
            const int jsize = this->info().jsize();
            const int ksize = this->info().ksize();
            const int istride = this->info().istride();
            const int jstride = this->info().jstride();
            const int kstride = this->info().kstride();

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

            vadv_kcache_kernel<<<m_blocked_execution.blocks(), m_blocked_execution.threads()>>>(ustage,
                upos,
                utens,
                utensstage,
                vstage,
                vpos,
                vtens,
                vtensstage,
                wstage,
                wpos,
                wtens,
                wtensstage,
                ccol,
                dcol,
                wcon,
                isize,
                jsize,
                ksize,
                istride,
                jstride,
                kstride);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace cuda
} // namespace backend
