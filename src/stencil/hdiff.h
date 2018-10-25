#pragma once

#include "except.h"
#include "real.h"
#include "stencil_execution.h"
#include <functional>
#include <random>

namespace stencil {

    template <class Allocator>
    class hdiff : public stencil_execution {
      public:
        static void register_arguments(arguments &args) { stencil_execution::register_arguments(args); }

        hdiff(const arguments_map &args)
            : stencil_execution(args), m_src(create_field<real, Allocator>()), m_dst(create_field<real, Allocator>()),
              m_coeff(create_field<real, Allocator>()) {
            if (info().halo() < 3)
                throw ERROR("halo must be larger than 2");
            std::mt19937 eng;
            std::uniform_real_distribution<real> dist(-1, 1);
            auto rand = [&](int, int, int) { return dist(eng); };
            m_src->fill(rand);
            m_dst->fill(rand);
            m_coeff->fill(rand);
        }

        bool verify() override {
            this->run();
            auto lap_field = create_field<real, std::allocator<real>>();
            auto flx_field = create_field<real, std::allocator<real>>();
            auto fly_field = create_field<real, std::allocator<real>>();
            auto ref_field = create_field<real, std::allocator<real>>();
            auto &src = *m_src;
            auto &dst = *m_dst;
            auto &coeff = *m_coeff;
            auto &lap = *lap_field;
            auto &flx = *flx_field;
            auto &fly = *fly_field;
            auto &ref = *ref_field;

            loop(
                [&](int i, int j, int k) {
                    lap(i, j, k) =
                        4 * src(i, j, k) - (src(i - 1, j, k) + src(i + 1, j, k) + src(i, j - 1, k) + src(i, j + 1, k));
                },
                2);

            loop(
                [&](int i, int j, int k) {
                    flx(i, j, k) = lap(i + 1, j, k) - lap(i, j, k);
                    if (flx(i, j, k) * (src(i + 1, j, k) - src(i, j, k)) > 0)
                        flx(i, j, k) = 0;
                },
                1);

            loop(
                [&](int i, int j, int k) {
                    fly(i, j, k) = lap(i, j + 1, k) - lap(i, j, k);
                    if (fly(i, j, k) * (src(i, j + 1, k) - src(i, j, k)) > 0)
                        fly(i, j, k) = 0;
                },
                1);

            return loop_check([&](int i, int j, int k) {
                ref(i, j, k) =
                    src(i, j, k) - coeff(i, j, k) * (flx(i, j, k) - flx(i - 1, j, k) + fly(i, j, k) - fly(i, j - 1, k));
                if (!real_equal(ref(i, j, k), dst(i, j, k))) {
#pragma omp critical
                    std::cout << i << " " << j << " " << k << ref(i, j, k) << " " << dst(i, j, k) << std::endl;
                }
                return real_equal(ref(i, j, k), dst(i, j, k));
            });
        }

        std::size_t touched_bytes() const override {
            const std::size_t isize = info().isize();
            const std::size_t jsize = info().jsize();
            const std::size_t ksize = info().ksize();
            return isize * jsize * ksize * 2 * sizeof(real);
        }

      protected:
        field_ptr<real, Allocator> m_src, m_dst, m_coeff;
    };
} // namespace stencil
