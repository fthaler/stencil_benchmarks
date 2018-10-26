#pragma once

#include "backend/cuda/allocator.h"
#include "backend/cuda/check.h"
#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace cuda {

        __global__ void hdiff_otf_kernel(real *__restrict__ dst,
            const real *__restrict__ src,
            const real *__restrict__ coeff,
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
                for (int j = jfirst - 2; j < jsize + 2; j += blockDim.y * gridDim.y) {
                    for (int i = ifirst - 2; i < isize + 2; i += blockDim.x * gridDim.x) {
                        const int index = i * istride + j * jstride + k * kstride;

                        real lap_ij = 4 * src[index] - src[index - istride] - src[index + istride] -
                                      src[index - jstride] - src[index + jstride];
                        real lap_imj = 4 * src[index - istride] - src[index - 2 * istride] - src[index] -
                                       src[index - istride - jstride] - src[index - istride + jstride];
                        real lap_ipj = 4 * src[index + istride] - src[index] - src[index + 2 * istride] -
                                       src[index + istride - jstride] - src[index + istride + jstride];
                        real lap_ijm = 4 * src[index - jstride] - src[index - istride - jstride] -
                                       src[index + istride - jstride] - src[index - 2 * jstride] - src[index];
                        real lap_ijp = 4 * src[index + jstride] - src[index - istride + jstride] -
                                       src[index + istride + jstride] - src[index] - src[index + 2 * jstride];

                        real flx_ij = lap_ipj - lap_ij;
                        flx_ij = flx_ij * (src[index + istride] - src[index]) > 0 ? 0 : flx_ij;

                        real flx_imj = lap_ij - lap_imj;
                        flx_imj = flx_imj * (src[index] - src[index - istride]) > 0 ? 0 : flx_imj;

                        real fly_ij = lap_ijp - lap_ij;
                        fly_ij = fly_ij * (src[index + jstride] - src[index]) > 0 ? 0 : fly_ij;

                        real fly_ijm = lap_ij - lap_ijm;
                        fly_ijm = fly_ijm * (src[index] - src[index - jstride]) > 0 ? 0 : fly_ijm;

                        dst[index] = src[index] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
                    }
                }
            }
        }

        class hdiff_otf : public stencil::hdiff<allocator<real>> {
          public:
            static void register_arguments(arguments &args) {
                stencil::hdiff<allocator<real>>::register_arguments(args);
                args.add({"i-blocks", "CUDA blocks in i-direction", "8"})
                    .add({"j-blocks", "CUDA blocks in j-direction", "8"})
                    .add({"k-blocks", "CUDA blocks in k-direction", "8"})
                    .add({"i-threads", "CUDA threads per block in i-direction", "8"})
                    .add({"j-threads", "CUDA threads per block in j-direction", "8"})
                    .add({"k-threads", "CUDA threads per block in k-direction", "8"});
            }

            hdiff_otf(const arguments_map &args)
                : stencil::hdiff<allocator<real>>(args), m_iblocks(args.get<int>("i-blocks")),
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

                const real *__restrict__ src = this->m_src->data();
                const real *__restrict__ coeff = this->m_coeff->data();
                real *__restrict__ dst = this->m_dst->data();

                dim3 blocks(m_iblocks, m_jblocks, m_kblocks);
                dim3 threads(m_ithreads, m_jthreads, m_kthreads);

                hdiff_otf_kernel<<<blocks, threads>>>(dst, src, coeff, isize, jsize, ksize, istride, jstride, kstride);

                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaDeviceSynchronize());
            }

          private:
            int m_iblocks, m_jblocks, m_kblocks;
            int m_ithreads, m_jthreads, m_kthreads;
        };

    } // namespace cuda
} // namespace backend
