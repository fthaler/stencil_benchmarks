#pragma once

#include "backend/cuda/check.h"
#include "except.h"
#include "stencil/basic.h"
#include "stencil/basic_functors.h"
#include <algorithm>
#include <array>
#include <utility>

namespace backend {
    namespace cuda {
        namespace stencil {

            template <class Functor>
            __global__ void basic_blocked_kernel(const Functor functor,
                const int isize,
                const int jsize,
                const int ksize,
                const int istride,
                const int jstride,
                const int kstride) {
                const int ifirst = blockIdx.x * blockDim.x + threadIdx.x;
                const int jfirst = blockIdx.y * blockDim.y + threadIdx.y;
                const int kfirst = blockIdx.z * blockDim.z + threadIdx.z;
                for (int k = kfirst; k < ksize; k += blockDim.z * gridDim.z) {
                    for (int j = jfirst; j < jsize; j += blockDim.y * gridDim.y) {
                        for (int i = ifirst; i < isize; i += blockDim.x * gridDim.x) {
                            const int index = i * istride + j * jstride + k * kstride;
                            functor(index);
                        }
                    }
                }
            }

            template <class Functor, class Allocator>
            class basic_base_blocked : public ::stencil::basic<Functor, Allocator> {
              public:
                static void register_arguments(arguments &args) {
                    ::stencil::basic<Functor, Allocator>::register_arguments(args);
                    args.add({"i-blocks", "CUDA blocks in i-direction", "8"})
                        .add({"j-blocks", "CUDA blocks in j-direction", "8"})
                        .add({"k-blocks", "CUDA blocks in k-direction", "8"})
                        .add({"i-threads", "CUDA threads per block in i-direction", "8"})
                        .add({"j-threads", "CUDA threads per block in j-direction", "8"})
                        .add({"k-threads", "CUDA threads per block in k-direction", "8"});
                }

                basic_base_blocked(const arguments_map &args)
                    : ::stencil::basic<Functor, Allocator>(args), m_iblocks(args.get<int>("i-blocks")),
                      m_jblocks(args.get<int>("j-blocks")), m_kblocks(args.get<int>("k-blocks")),
                      m_ithreads(args.get<int>("i-threads")), m_jthreads(args.get<int>("j-threads")),
                      m_kthreads(args.get<int>("k-threads")) {
                    if (m_iblocks <= 0)
                        throw ERROR("invalid i-blocks");
                    if (m_jblocks <= 0)
                        throw ERROR("invalid j-blocks");
                    if (m_kblocks <= 0)
                        throw ERROR("invalid k-blocks");
                    if (m_ithreads <= 0)
                        throw ERROR("invalid i-threads");
                    if (m_jthreads <= 0)
                        throw ERROR("invalid j-threads");
                    if (m_kthreads <= 0)
                        throw ERROR("invalid k-threads");
                }

                void run() override {
                    const int isize = this->info().isize();
                    const int jsize = this->info().jsize();
                    const int ksize = this->info().ksize();
                    const int istride = this->info().istride();
                    const int jstride = this->info().jstride();
                    const int kstride = this->info().kstride();
                    Functor functor(this->info(), this->m_src->data(), this->m_dst->data());

                    dim3 blocks(m_iblocks, m_jblocks, m_kblocks);
                    dim3 threads(m_ithreads, m_jthreads, m_kthreads);
                    basic_blocked_kernel<<<blocks, threads>>>(functor, isize, jsize, ksize, istride, jstride, kstride);
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

              private:
                int m_iblocks, m_jblocks, m_kblocks;
                int m_ithreads, m_jthreads, m_kthreads;
            };

            template <class Allocator>
            using basic_copy_blocked = basic_base_blocked<::stencil::copy_functor, Allocator>;

            template <class Allocator>
            using basic_avgi_blocked = basic_base_blocked<::stencil::avgi_functor, Allocator>;
            template <class Allocator>
            using basic_avgj_blocked = basic_base_blocked<::stencil::avgj_functor, Allocator>;
            template <class Allocator>
            using basic_avgk_blocked = basic_base_blocked<::stencil::avgk_functor, Allocator>;

            template <class Allocator>
            using basic_lapij_blocked = basic_base_blocked<::stencil::lapij_functor, Allocator>;
            template <class Allocator>
            using basic_lapik_blocked = basic_base_blocked<::stencil::lapik_functor, Allocator>;
            template <class Allocator>
            using basic_lapjk_blocked = basic_base_blocked<::stencil::lapjk_functor, Allocator>;
            template <class Allocator>
            using basic_lapijk_blocked = basic_base_blocked<::stencil::lapijk_functor, Allocator>;
        } // namespace stencil
    }     // namespace cuda
} // namespace backend
