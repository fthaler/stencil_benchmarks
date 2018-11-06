#include "backend/cuda/block_index.h"
#include "backend/cuda/check.h"
#include "backend/cuda/hdiff_otf_fillcache.h"
#include "except.h"

namespace backend {
    namespace cuda {
        __global__ void hdiff_otf_fillcache_kernel(real *__restrict__ dst,
            const real *__restrict__ src,
            const real *__restrict__ coeff,
            const int isize,
            const int jsize,
            const int ksize,
            const int istride,
            const int jstride,
            const int kstride) {
            using block_index_t = hdiff_otf_fillcache::blocked_execution_t::block_index_t;

            const block_index_t bidx(isize, jsize, ksize);
            constexpr int block_halo = block_index_t::halo();

            constexpr int icachestride = 1;
            const int jcachestride = (bidx.iblocksize + 2 * block_halo) * icachestride;
            const int kcachestride = (bidx.jblocksize + 2 * block_halo) * jcachestride;

            extern __shared__ char smem[];

            real *__restrict__ inc =
                reinterpret_cast<real *>(&smem[0]) + icachestride * block_halo + jcachestride * block_halo;

            const int cacheindex = bidx.iblock * icachestride + bidx.jblock * jcachestride + bidx.kblock * kcachestride;
            const int index = bidx.i * istride + bidx.j * jstride + bidx.k * kstride;

            if (bidx.in_block_or_halo())
                inc[cacheindex] = src[index];

            __syncthreads();

            if (bidx.in_block()) {
                real lap_ij = 4 * inc[cacheindex] - inc[cacheindex - icachestride] - inc[cacheindex + icachestride] -
                              inc[cacheindex - jcachestride] - inc[cacheindex + jcachestride];
                real lap_imj = 4 * inc[cacheindex - icachestride] - inc[cacheindex - 2 * icachestride] -
                               inc[cacheindex] - inc[cacheindex - icachestride - jcachestride] -
                               inc[cacheindex - icachestride + jcachestride];
                real lap_ipj = 4 * inc[cacheindex + icachestride] - inc[cacheindex] -
                               inc[cacheindex + 2 * icachestride] - inc[cacheindex + icachestride - jcachestride] -
                               inc[cacheindex + icachestride + jcachestride];
                real lap_ijm = 4 * inc[cacheindex - jcachestride] - inc[cacheindex - icachestride - jcachestride] -
                               inc[cacheindex + icachestride - jcachestride] - inc[cacheindex - 2 * jcachestride] -
                               inc[cacheindex];
                real lap_ijp = 4 * inc[cacheindex + jcachestride] - inc[cacheindex - icachestride + jcachestride] -
                               inc[cacheindex + icachestride + jcachestride] - inc[cacheindex] -
                               inc[cacheindex + 2 * jcachestride];

                real flx_ij = lap_ipj - lap_ij;
                flx_ij = flx_ij * (inc[cacheindex + icachestride] - inc[cacheindex]) > 0 ? 0 : flx_ij;

                real flx_imj = lap_ij - lap_imj;
                flx_imj = flx_imj * (inc[cacheindex] - inc[cacheindex - icachestride]) > 0 ? 0 : flx_imj;

                real fly_ij = lap_ijp - lap_ij;
                fly_ij = fly_ij * (inc[cacheindex + jcachestride] - inc[cacheindex]) > 0 ? 0 : fly_ij;

                real fly_ijm = lap_ij - lap_ijm;
                fly_ijm = fly_ijm * (inc[cacheindex] - inc[cacheindex - jcachestride]) > 0 ? 0 : fly_ijm;

                dst[index] = inc[cacheindex] - coeff[index] * (flx_ij - flx_imj + fly_ij - fly_ijm);
            }
        }

        void hdiff_otf_fillcache::register_arguments(arguments &args) {
            stencil::hdiff<allocator<real>>::register_arguments(args);
            blocked_execution_t::register_arguments(args);
        }

        hdiff_otf_fillcache::hdiff_otf_fillcache(const arguments_map &args)
            : stencil::hdiff<allocator<real>>(args), m_blocked_execution(args) {}

        void hdiff_otf_fillcache::run() {
            const int isize = this->info().isize();
            const int jsize = this->info().jsize();
            const int ksize = this->info().ksize();
            const int istride = this->info().istride();
            const int jstride = this->info().jstride();
            const int kstride = this->info().kstride();

            const real *__restrict__ src = this->m_src->data();
            const real *__restrict__ coeff = this->m_coeff->data();
            real *__restrict__ dst = this->m_dst->data();

            hdiff_otf_fillcache_kernel<<<m_blocked_execution.blocks(),
                m_blocked_execution.threads(),
                m_blocked_execution.blocksize_with_halo() * sizeof(real)>>>(
                dst, src, coeff, isize, jsize, ksize, istride, jstride, kstride);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace cuda
} // namespace backend
