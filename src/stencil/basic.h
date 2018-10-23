#pragma once

#include "real.h"
#include "stencil_execution.h"
#include <random>

namespace stencil {

    template <class Functor, class Allocator>
    class basic : public stencil_execution {
      public:
        static void register_arguments(arguments &args) { stencil_execution::register_arguments(args); }

        basic(const arguments_map &args)
            : stencil_execution(args), m_src(create_field<real, Allocator>()), m_dst(create_field<real, Allocator>()) {
            std::mt19937 eng;
            std::uniform_real_distribution<real> dist;
            m_src->fill([&](int, int, int) { return dist(eng); });
            m_dst->fill([&](int, int, int) { return dist(eng); });
        }

        bool verify() override {
            this->run();
            auto ref = create_field<real, std::allocator<real>>();
            Functor functor(info(), m_src->data(), ref->data());
            return loop_check([&](int i, int j, int k) {
                functor(info().index(i, j, k));
                return (*ref)(i, j, k) == (*m_dst)(i, j, k);
            });
        }

        std::size_t touched_bytes() const override {
            const std::size_t isize = info().isize();
            const std::size_t jsize = info().jsize();
            const std::size_t ksize = info().ksize();
            return isize * jsize * ksize * 2 * sizeof(real);
        };

      protected:
        field_ptr<real, Allocator> m_src, m_dst;
    };

} // namespace stencil
