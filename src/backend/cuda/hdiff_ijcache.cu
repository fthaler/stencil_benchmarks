#include "backend/cuda/block_index.h"
#include "backend/cuda/check.h"
#include "backend/cuda/hdiff_ijcache.h"
#include "except.h"

namespace backend {
    namespace cuda {
        __global__ void hdiff_ijcache_kernel(real *__restrict__ dst,
            const real *__restrict__ src,
            const real *__restrict__ coeff,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            using block_index_t = hdiff_ijcache::blocked_execution_t::block_index_t;

            const block_index_t bidx(isize, jsize, ksize);
            constexpr int block_halo = block_index_t::halo();

            constexpr int icachestride = 1;
            const int jcachestride = (bidx.iblocksize + 2 * block_halo) * icachestride;
            const int kcachestride = (bidx.jblocksize + 2 * block_halo) * jcachestride;

            extern __shared__ char smem[];

            real *__restrict__ smem_real = reinterpret_cast<real *>(&smem[0]);
            const int cache_size = bidx.blocksize_with_halo();
            const int cache_data_offset = icachestride * block_halo + jcachestride * block_halo;
            real *__restrict__ lap = smem_real + cache_data_offset;
            real *__restrict__ flx = smem_real + cache_size + cache_data_offset;
            real *__restrict__ fly = smem_real + 2 * cache_size + cache_data_offset;

            const int cacheindex = bidx.iblock * icachestride + bidx.jblock * jcachestride + bidx.kblock * kcachestride;
            const int index = bidx.i * istride + bidx.j * jstride + bidx.k * kstride;

            if (bidx.in_block_or_halo()) {
                lap[cacheindex] = 4 * src[index] - src[index - istride] - src[index + istride] - src[index - jstride] -
                                  src[index + jstride];
            }

            __syncthreads();

            if (bidx.in_block_or_halo(1, 0, 1, 0)) {
                flx[cacheindex] = lap[cacheindex + icachestride] - lap[cacheindex];
                if (flx[cacheindex] * (src[index + istride] - src[index]) > 0)
                    flx[cacheindex] = 0;

                fly[cacheindex] = lap[cacheindex + jcachestride] - lap[cacheindex];
                if ((fly[cacheindex] > 0) == ((src[index + jstride] - src[index]) > 0))
                    fly[cacheindex] = 0;
            }

            __syncthreads();

            if (bidx.in_block()) {
                dst[index] = src[index] - coeff[index] * (flx[cacheindex] - flx[cacheindex - icachestride] +
                                                             fly[cacheindex] - fly[cacheindex - jcachestride]);
            }
        }

        void hdiff_ijcache::register_arguments(arguments &args) {
            stencil::hdiff<allocator<real>>::register_arguments(args);
            blocked_execution_t::register_arguments(args);
        }

        hdiff_ijcache::hdiff_ijcache(const arguments_map &args)
            : stencil::hdiff<allocator<real>>(args), m_blocked_execution(args) {
            CUDA_CHECK(cudaFuncSetCacheConfig(hdiff_ijcache_kernel, cudaFuncCachePreferL1));
        }

        void hdiff_ijcache::run() {
            const int isize = this->info().isize();
            const int jsize = this->info().jsize();
            const int ksize = this->info().ksize();
            const int istride = this->info().istride();
            const int jstride = this->info().jstride();
            const int kstride = this->info().kstride();

            const real *__restrict__ src = this->m_src->data();
            const real *__restrict__ coeff = this->m_coeff->data();
            real *__restrict__ dst = this->m_dst->data();

            hdiff_ijcache_kernel<<<m_blocked_execution.blocks(),
                m_blocked_execution.threads(),
                3 * m_blocked_execution.blocksize_with_halo() * sizeof(real)>>>(
                dst, src, coeff, isize, jsize, ksize, istride, jstride, kstride);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace cuda
} // namespace backend
