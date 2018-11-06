#include "backend/cuda/blocked_execution.h"
#include "backend/cuda/check.h"
#include "backend/cuda/hdiff_otf.h"
#include "except.h"

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
            using block_index_t = hdiff_otf::blocked_execution_t::block_index_t;

            const block_index_t bidx(isize, jsize, ksize);
            const int index = bidx.i * istride + bidx.j * jstride + bidx.k * kstride;

            if (bidx.in_block()) {
                real lap_ij = 4 * src[index] - src[index - istride] - src[index + istride] - src[index - jstride] -
                              src[index + jstride];
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

        void hdiff_otf::register_arguments(arguments &args) {
            stencil::hdiff<allocator<real>>::register_arguments(args);
            blocked_execution_t::register_arguments(args);
        }

        hdiff_otf::hdiff_otf(const arguments_map &args)
            : stencil::hdiff<allocator<real>>(args), m_blocked_execution(args) {}

        void hdiff_otf::run() {
            const int isize = this->info().isize();
            const int jsize = this->info().jsize();
            const int ksize = this->info().ksize();
            const int istride = this->info().istride();
            const int jstride = this->info().jstride();
            const int kstride = this->info().kstride();

            const real *__restrict__ src = this->m_src->data();
            const real *__restrict__ coeff = this->m_coeff->data();
            real *__restrict__ dst = this->m_dst->data();

            hdiff_otf_kernel<<<m_blocked_execution.blocks(), m_blocked_execution.threads()>>>(
                dst, src, coeff, isize, jsize, ksize, istride, jstride, kstride);

            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaDeviceSynchronize());
        }

    } // namespace cuda
} // namespace backend
