#pragma once

#include "backend/cuda/check.h"
#include "except.h"
#include "stencil/basic.h"
#include "stencil/basic_functors.h"

namespace backend {
    namespace cuda {

        template <class Functor>
        __global__ void basic_1d_kernel(const Functor functor, const int ilast) {
            const int ifirst = blockIdx.x * blockDim.x + threadIdx.x;
            for (int i = ifirst; i <= ilast; i += blockDim.x * gridDim.x) {
                functor(i);
            }
        }

        template <class Functor, class Allocator>
        class basic_base_1d : public stencil::basic<Functor, Allocator> {
          public:
            using stencil::basic<Functor, Allocator>::basic;

            static void register_arguments(arguments &args) {
                stencil::basic<Functor, Allocator>::register_arguments(args);
                args.add({"cuda-blocks", "CUDA blocks", "512"}).add({"cuda-threads", "CUDA threads per block", "512"});
            }

            basic_base_1d(const arguments_map &args)
                : stencil::basic<Functor, Allocator>(args), m_blocks(args.get<int>("cuda-blocks")),
                  m_threads(args.get<int>("cuda-threads")) {
                if (m_blocks < 0)
                    throw ERROR("invalid CUDA block count");
                if (m_threads < 0)
                    throw ERROR("invalid CUDA thread count");
            }

            void run() override {
                const int ilast = this->info().last_index();
                Functor functor(this->info(), this->m_src->data(), this->m_dst->data());

                basic_1d_kernel<<<m_blocks, m_threads>>>(functor, ilast);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

          private:
            int m_blocks, m_threads;
        };

        template <class Allocator>
        using basic_copy_1d = basic_base_1d<stencil::copy_functor, Allocator>;

        template <class Allocator>
        using basic_avgi_1d = basic_base_1d<stencil::avgi_functor, Allocator>;
        template <class Allocator>
        using basic_avgj_1d = basic_base_1d<stencil::avgj_functor, Allocator>;
        template <class Allocator>
        using basic_avgk_1d = basic_base_1d<stencil::avgk_functor, Allocator>;

        template <class Allocator>
        using basic_lapij_1d = basic_base_1d<stencil::lapij_functor, Allocator>;
        template <class Allocator>
        using basic_lapik_1d = basic_base_1d<stencil::lapik_functor, Allocator>;
        template <class Allocator>
        using basic_lapjk_1d = basic_base_1d<stencil::lapjk_functor, Allocator>;
        template <class Allocator>
        using basic_lapijk_1d = basic_base_1d<stencil::lapijk_functor, Allocator>;

    } // namespace cuda
} // namespace backend
