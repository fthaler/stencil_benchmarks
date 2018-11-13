#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "real.h"
#include "stencil_execution.h"

namespace stencil {

    template <class Allocator>
    class vadv : public stencil_execution {
      public:
        static void register_arguments(arguments &args) { stencil_execution::register_arguments(args); }

        vadv(const arguments_map &args);

        bool verify() override;

        std::size_t touched_bytes() const override {
            const std::size_t isize = info().isize();
            const std::size_t jsize = info().jsize();
            const std::size_t ksize = info().ksize();
            // TODO: better estimate
            return isize * jsize * ksize * 16 * sizeof(real);
        }

      protected:
        static constexpr real dtr_stage = 3.0 / 20.0;
        static constexpr real beta_v = 0;
        static constexpr real bet_m = 0.5 * (1.0 - beta_v);
        static constexpr real bet_p = 0.5 * (1.0 + beta_v);

      protected:
        std::function<real(int, int, int)> ic_fill(
            real offset1, real offset2, real base1, real base2, real ispread, real jspread) const;

        static std::function<real(int, int, int)> nan_fill();

        field_ptr<real, Allocator> m_ustage, m_upos, m_utens, m_utensstage, m_vstage, m_vpos, m_vtens, m_vtensstage,
            m_wstage, m_wpos, m_wtens, m_wtensstage, m_ccol, m_dcol, m_wcon, m_datacol;
    };

    template <class Allocator>
    vadv<Allocator>::vadv(const arguments_map &args)
        : stencil_execution(args), m_ustage(create_field<real, Allocator>()), m_upos(create_field<real, Allocator>()),
          m_utens(create_field<real, Allocator>()), m_utensstage(create_field<real, Allocator>()),
          m_vstage(create_field<real, Allocator>()), m_vpos(create_field<real, Allocator>()),
          m_vtens(create_field<real, Allocator>()), m_vtensstage(create_field<real, Allocator>()),
          m_wstage(create_field<real, Allocator>()), m_wpos(create_field<real, Allocator>()),
          m_wtens(create_field<real, Allocator>()), m_wtensstage(create_field<real, Allocator>()),
          m_ccol(create_field<real, Allocator>()), m_dcol(create_field<real, Allocator>()),
          m_wcon(create_field<real, Allocator>()), m_datacol(create_field<real, Allocator>()) {
        m_ustage->fill(ic_fill(2.2, 1.5, 0.95, 1.18, 18.4, 20.3));
        m_upos->fill(ic_fill(3.4, 0.7, 1.07, 1.51, 1.4, 2.3));
        m_utens->fill(ic_fill(7.4, 4.3, 1.17, 0.91, 1.4, 2.3));
        m_utensstage->fill(ic_fill(3.2, 2.5, 0.95, 1.18, 18.4, 20.3));

        m_vstage->fill(ic_fill(2.3, 1.5, 0.95, 1.14, 18.4, 20.3));
        m_vpos->fill(ic_fill(3.3, 0.7, 1.07, 1.71, 1.4, 2.3));
        m_vtens->fill(ic_fill(7.3, 4.3, 1.17, 0.71, 1.4, 2.3));
        m_vtensstage->fill(ic_fill(3.3, 2.4, 0.95, 1.18, 18.4, 20.3));

        m_wstage->fill(ic_fill(2.3, 1.5, 0.95, 1.14, 18.4, 20.3));
        m_wpos->fill(ic_fill(3.3, 0.7, 1.07, 1.71, 1.4, 2.3));
        m_wtens->fill(ic_fill(7.3, 4.3, 1.17, 0.71, 1.4, 2.3));
        m_wtensstage->fill(ic_fill(3.3, 2.4, 0.95, 1.18, 18.4, 20.3));

        m_wcon->fill(ic_fill(1.3, 0.3, 0.87, 1.14, 1.4, 2.3));

        m_ccol->fill(nan_fill());
        m_dcol->fill(nan_fill());
        m_datacol->fill(nan_fill());
    }

    template <class Allocator>
    bool vadv<Allocator>::verify() {
        auto utensstage_ref = create_field<real, std::allocator<real>>();
        auto vtensstage_ref = create_field<real, std::allocator<real>>();
        auto wtensstage_ref = create_field<real, std::allocator<real>>();

        std::copy(this->m_utensstage->begin(), this->m_utensstage->end(), utensstage_ref->begin());
        std::copy(this->m_vtensstage->begin(), this->m_vtensstage->end(), vtensstage_ref->begin());
        std::copy(this->m_wtensstage->begin(), this->m_wtensstage->end(), wtensstage_ref->begin());

        this->run();

        const int isize = info().isize();
        const int jsize = info().jsize();
        const int ksize = info().ksize();
        const int istride = info().istride();
        const int jstride = info().jstride();
        const int kstride = info().kstride();

        auto backward_sweep =
            [ksize, istride, jstride, kstride](
                int i, int j, real *ccol, const real *dcol, real *datacol, const real *upos, real *utensstage) {
                // k maximum
                {
                    const int k = ksize - 1;
                    const int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }

                // k body
                for (int k = ksize - 2; k >= 0; --k) {
                    const int index = i * istride + j * jstride + k * kstride;
                    datacol[index] = dcol[index] - ccol[index] * datacol[index + kstride];
                    ccol[index] = datacol[index];
                    utensstage[index] = dtr_stage * (datacol[index] - upos[index]);
                }
            };
        auto forward_sweep = [ksize, istride, jstride, kstride](int i,
                                 int j,
                                 int ishift,
                                 int jshift,
                                 real *ccol,
                                 real *dcol,
                                 const real *wcon,
                                 const real *ustage,
                                 const real *upos,
                                 const real *utens,
                                 const real *utensstage) {
            // k minimum
            {
                const int k = 0;
                const int index = i * istride + j * jstride + k * kstride;
                real gcv =
                    real(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                real cs = gcv * bet_m;

                ccol[index] = gcv * bet_p;
                real bcol = dtr_stage - ccol[index];

                real correction_term = -cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / bcol;
                ccol[index] = ccol[index] * divided;
                dcol[index] = dcol[index] * divided;
            }

            // k body
            for (int k = 1; k < ksize - 1; ++k) {
                const int index = i * istride + j * jstride + k * kstride;
                real gav = real(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);
                real gcv =
                    real(0.25) * (wcon[index + ishift * istride + jshift * jstride + kstride] + wcon[index + kstride]);

                real as = gav * bet_m;
                real cs = gcv * bet_m;

                real acol = gav * bet_p;
                ccol[index] = gcv * bet_p;
                real bcol = dtr_stage - acol - ccol[index];

                real correction_term =
                    -as * (ustage[index - kstride] - ustage[index]) - cs * (ustage[index + kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / (bcol - ccol[index - kstride] * acol);
                ccol[index] = ccol[index] * divided;
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }

            // k maximum
            {
                const int k = ksize - 1;
                const int index = i * istride + j * jstride + k * kstride;
                real gav = real(-0.25) * (wcon[index + ishift * istride + jshift * jstride] + wcon[index]);

                real as = gav * bet_m;

                real acol = gav * bet_p;
                real bcol = dtr_stage - acol;

                real correction_term = -as * (ustage[index - kstride] - ustage[index]);
                dcol[index] = dtr_stage * upos[index] + utens[index] + utensstage[index] + correction_term;

                real divided = real(1.0) / (bcol - ccol[index - kstride] * acol);
                dcol[index] = (dcol[index] - dcol[index - kstride] * acol) * divided;
            }
        };

        m_ccol->fill(nan_fill());
        m_dcol->fill(nan_fill());
        m_datacol->fill(nan_fill());

// generate u
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i,
                    j,
                    1,
                    0,
                    m_ccol->data(),
                    m_dcol->data(),
                    m_wcon->data(),
                    m_ustage->data(),
                    m_upos->data(),
                    m_utens->data(),
                    utensstage_ref->data());
                backward_sweep(
                    i, j, m_ccol->data(), m_dcol->data(), m_datacol->data(), m_upos->data(), utensstage_ref->data());
            }

// generate v
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i,
                    j,
                    0,
                    1,
                    m_ccol->data(),
                    m_dcol->data(),
                    m_wcon->data(),
                    m_vstage->data(),
                    m_vpos->data(),
                    m_vtens->data(),
                    vtensstage_ref->data());
                backward_sweep(
                    i, j, m_ccol->data(), m_dcol->data(), m_datacol->data(), m_vpos->data(), vtensstage_ref->data());
            }

// generate w
#pragma omp parallel for collapse(2)
        for (int j = 0; j < jsize; ++j)
            for (int i = 0; i < isize; ++i) {
                forward_sweep(i,
                    j,
                    0,
                    0,
                    m_ccol->data(),
                    m_dcol->data(),
                    m_wcon->data(),
                    m_wstage->data(),
                    m_wpos->data(),
                    m_wtens->data(),
                    wtensstage_ref->data());
                backward_sweep(
                    i, j, m_ccol->data(), m_dcol->data(), m_datacol->data(), m_wpos->data(), wtensstage_ref->data());
            }

        auto eq = [](real a, real b) {
            real diff = std::abs(a - b);
            a = std::abs(a);
            b = std::abs(b);
            return diff <= (a > b ? a : b) * 1e-3;
        };

        return loop_check([&](int i, int j, int k) {
            bool usuccess = eq((*m_utensstage)(i, j, k), (*utensstage_ref)(i, j, k));
            bool vsuccess = eq((*m_vtensstage)(i, j, k), (*vtensstage_ref)(i, j, k));
            bool wsuccess = eq((*m_wtensstage)(i, j, k), (*wtensstage_ref)(i, j, k));
            return usuccess && vsuccess && wsuccess;
        });
    }

    template <class Allocator>
    std::function<real(int, int, int)> vadv<Allocator>::ic_fill(
        real offset1, real offset2, real base1, real base2, real ispread, real jspread) const {
        real di = 1.0 / info().isize();
        real dj = 1.0 / info().jsize();
        real dk = 1.0 / info().ksize();
        return [=](int i, int j, int k) {
            return offset1 + base1 *
                                 (offset2 + std::cos(M_PI * (ispread * i * di + ispread * j * dj)) +
                                     base2 * std::sin(2 * M_PI * (ispread * i * di + jspread * j * dj) * k * dk)) /
                                 4.0;
        };
    }

    template <class Allocator>
    std::function<real(int, int, int)> vadv<Allocator>::nan_fill() {
        return [](int, int, int) { return std::numeric_limits<real>::signaling_NaN(); };
    }

} // namespace stencil
