#pragma once

#include "except.h"
#include "stencil/basic.h"
#include "stencil/basic_functors.h"
#include <algorithm>
#include <array>
#include <utility>

namespace backend {
    namespace openmp {

        template <class Functor, class Allocator>
        class basic_base_blocked : public stencil::basic<Functor, Allocator> {
          public:
            static void register_arguments(arguments &args) {
                stencil::basic<Functor, Allocator>::register_arguments(args);
                args.add({"i-blocksize", "block size in i-direction", "8"})
                    .add({"j-blocksize", "block size in j-direction", "8"})
                    .add({"k-blocksize", "block size in k-direction", "8"});
            }

            basic_base_blocked(const arguments_map &args)
                : stencil::basic<Functor, Allocator>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")), m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");
                if (m_kblocksize <= 0)
                    throw ERROR("invalid k-blocksize");
            }

            void run() override {
                const int isize = this->info().isize();
                const int jsize = this->info().jsize();
                const int ksize = this->info().ksize();
                const int istride = this->info().istride();
                const int jstride = this->info().jstride();
                const int kstride = this->info().kstride();
                Functor functor(this->info(), this->m_src->data(), this->m_dst->data());

                std::array<std::tuple<int, int, int>, 3> ssb{std::make_tuple(istride, isize, m_iblocksize),
                    std::make_tuple(jstride, jsize, m_jblocksize),
                    std::make_tuple(kstride, ksize, m_kblocksize)};

                std::sort(ssb.begin(), ssb.end());

                int inner_stride, inner_size, inner_blocksize;
                int middle_stride, middle_size, middle_blocksize;
                int outer_stride, outer_size, outer_blocksize;

                std::tie(inner_stride, inner_size, inner_blocksize) = ssb[0];
                std::tie(middle_stride, middle_size, middle_blocksize) = ssb[1];
                std::tie(outer_stride, outer_size, outer_blocksize) = ssb[2];

                if (inner_stride != 1)
                    throw ERROR("data must be contiguous along one axis");

#pragma omp parallel for collapse(3)
                for (int outer_ib = 0; outer_ib < outer_size; outer_ib += outer_blocksize) {
                    for (int middle_ib = 0; middle_ib < middle_size; middle_ib += middle_blocksize) {
                        for (int inner_ib = 0; inner_ib < inner_size; inner_ib += inner_blocksize) {
                            const int outer_imax =
                                outer_ib + outer_blocksize <= outer_size ? outer_ib + outer_blocksize : outer_size;
                            const int middle_imax = middle_ib + middle_blocksize <= middle_size
                                                        ? middle_ib + middle_blocksize
                                                        : middle_size;
                            const int inner_imax =
                                inner_ib + inner_blocksize <= inner_size ? inner_ib + inner_blocksize : inner_size;

                            int index = outer_ib * outer_stride + middle_ib * middle_stride + inner_ib * inner_stride;

                            for (int outer_i = outer_ib; outer_i < outer_imax; ++outer_i) {
                                for (int middle_i = middle_ib; middle_i < middle_imax; ++middle_i) {
#pragma omp simd
#pragma vector nontemporal
                                    for (int inner_i = inner_ib; inner_i < inner_imax; ++inner_i) {
                                        functor(index);
                                        ++index;
                                    }

                                    index += middle_stride - (inner_imax - inner_ib);
                                }
                                index += outer_stride - (middle_imax - middle_ib) * middle_stride;
                            }
                        }
                    }
                }
            }

          private:
            int m_iblocksize, m_jblocksize, m_kblocksize;
        };

        template <class Allocator>
        using basic_copy_blocked = basic_base_blocked<stencil::copy_functor, Allocator>;

        template <class Allocator>
        using basic_avgi_blocked = basic_base_blocked<stencil::avgi_functor, Allocator>;
        template <class Allocator>
        using basic_avgj_blocked = basic_base_blocked<stencil::avgj_functor, Allocator>;
        template <class Allocator>
        using basic_avgk_blocked = basic_base_blocked<stencil::avgk_functor, Allocator>;

        template <class Allocator>
        using basic_lapij_blocked = basic_base_blocked<stencil::lapij_functor, Allocator>;
        template <class Allocator>
        using basic_lapik_blocked = basic_base_blocked<stencil::lapik_functor, Allocator>;
        template <class Allocator>
        using basic_lapjk_blocked = basic_base_blocked<stencil::lapjk_functor, Allocator>;
        template <class Allocator>
        using basic_lapijk_blocked = basic_base_blocked<stencil::lapijk_functor, Allocator>;

    } // namespace openmp
} // namespace backend
