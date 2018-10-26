#pragma once

#include "backend/cuda/check.h"
#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>

namespace backend {
    namespace cuda {
        namespace stencil {
            __global__ void hdiff_naive_lap_kernel(real *__restrict__ lap,
                const real *__restrict__ src,
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

                            lap[index] = 4 * src[index] - (src[index - istride] + src[index + istride] +
                                                              src[index - jstride] + src[index + jstride]);
                        }
                    }
                }
            }

            __global__ void hdiff_naive_flx_kernel(real *__restrict__ flx,
                const real *__restrict__ src,
                const real *__restrict__ lap,
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
                        for (int i = ifirst - 1; i < isize + 1; i += blockDim.x * gridDim.x) {
                            const int index = i * istride + j * jstride + k * kstride;

                            flx[index] = lap[index + istride] - lap[index];
                            if (flx[index] * (src[index + istride] - src[index]) > 0)
                                flx[index] = 0;
                        }
                    }
                }
            }

            __global__ void hdiff_naive_fly_kernel(real *__restrict__ fly,
                const real *__restrict__ src,
                const real *__restrict__ lap,
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
                    for (int j = jfirst - 1; j < jsize + 1; j += blockDim.y * gridDim.y) {
                        for (int i = ifirst; i < isize; i += blockDim.x * gridDim.x) {
                            const int index = i * istride + j * jstride + k * kstride;

                            fly[index] = lap[index + jstride] - lap[index];
                            if (fly[index] * (src[index + jstride] - src[index]) > 0)
                                fly[index] = 0;
                        }
                    }
                }
            }
            __global__ void hdiff_naive_dst_kernel(real *__restrict__ dst,
                const real *__restrict__ src,
                const real *__restrict__ coeff,
                const real *__restrict__ flx,
                const real *__restrict__ fly,
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

                            dst[index] = src[index] - coeff[index] * (flx[index] - flx[index - istride] + fly[index] -
                                                                         fly[index - jstride]);
                        }
                    }
                }
            }

            template <class Allocator>
            class hdiff_naive : public ::stencil::hdiff<Allocator> {
              public:
                static void register_arguments(arguments &args) {
                    ::stencil::hdiff<Allocator>::register_arguments(args);
                    args.add({"i-blocks", "CUDA blocks in i-direction", "8"})
                        .add({"j-blocks", "CUDA blocks in j-direction", "8"})
                        .add({"k-blocks", "CUDA blocks in k-direction", "8"})
                        .add({"i-threads", "CUDA threads per block in i-direction", "8"})
                        .add({"j-threads", "CUDA threads per block in j-direction", "8"})
                        .add({"k-threads", "CUDA threads per block in k-direction", "8"});
                }

                hdiff_naive(const arguments_map &args)
                    : ::stencil::hdiff<Allocator>(args), m_lap(this->template create_field<real, Allocator>()),
                      m_flx(this->template create_field<real, Allocator>()),
                      m_fly(this->template create_field<real, Allocator>()), m_iblocks(args.get<int>("i-blocks")),
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
                    real *__restrict__ lap = m_lap->data();
                    real *__restrict__ flx = m_flx->data();
                    real *__restrict__ fly = m_fly->data();
                    real *__restrict__ dst = this->m_dst->data();

                    dim3 blocks(m_iblocks, m_jblocks, m_kblocks);
                    dim3 threads(m_ithreads, m_jthreads, m_kthreads);

                    hdiff_naive_lap_kernel<<<blocks, threads>>>(
                        lap, src, isize, jsize, ksize, istride, jstride, kstride);
                    hdiff_naive_flx_kernel<<<blocks, threads>>>(
                        flx, src, lap, isize, jsize, ksize, istride, jstride, kstride);
                    hdiff_naive_fly_kernel<<<blocks, threads>>>(
                        fly, src, lap, isize, jsize, ksize, istride, jstride, kstride);
                    hdiff_naive_dst_kernel<<<blocks, threads>>>(
                        dst, src, coeff, flx, fly, isize, jsize, ksize, istride, jstride, kstride);

                    CUDA_CHECK(cudaGetLastError());
                    CUDA_CHECK(cudaDeviceSynchronize());
                }

              private:
                field_ptr<real, Allocator> m_lap, m_flx, m_fly;
                int m_iblocks, m_jblocks, m_kblocks;
                int m_ithreads, m_jthreads, m_kthreads;
            };

        } // namespace stencil
    }     // namespace cuda
} // namespace backend
