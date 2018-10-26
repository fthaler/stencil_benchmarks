#pragma once

#include "except.h"
#include "real.h"
#include "stencil/hdiff.h"
#include <algorithm>
#include <cassert>
#include <omp.h>

namespace backend {
    namespace openmp {

        template <class Allocator>
        class hdiff_ijcache_blocked : public stencil::hdiff<Allocator> {
          public:
            static void register_arguments(arguments &args) {
                stencil::hdiff<Allocator>::register_arguments(args);
                args.add({"i-blocksize", "block size in i-direction", "32"})
                    .add({"j-blocksize", "block size in j-direction", "32"});
            }

            hdiff_ijcache_blocked(const arguments_map &args)
                : stencil::hdiff<Allocator>(args), m_iblocksize(args.get<int>("i-blocksize")),
                  m_jblocksize(args.get<int>("j-blocksize")), m_cache_info(m_iblocksize,   // cache i-size
                                                                  m_jblocksize,            // cache j-size
                                                                  1,                       // cache k-size
                                                                  2,                       // cache i-layout
                                                                  1,                       // cache j-layout
                                                                  0,                       // cache k-layout
                                                                  2,                       // cache halo
                                                                  this->info().alignment() // cache alignment
                                                              ) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");

                int max_threads = omp_get_max_threads();
                m_lap_caches.resize(max_threads);
                for (auto &cache : m_lap_caches)
                    cache.resize(m_cache_info.storage_size());
                m_flx_caches.resize(max_threads);
                for (auto &cache : m_flx_caches)
                    cache.resize(m_cache_info.storage_size());
                m_fly_caches.resize(max_threads);
                for (auto &cache : m_fly_caches)
                    cache.resize(m_cache_info.storage_size());
            }

            void run() override {
                const int isize = this->info().isize();
                const int jsize = this->info().jsize();
                const int ksize = this->info().ksize();
                if (this->info().istride() != 1)
                    throw ERROR("i-stride must be 1");
                constexpr int istride = 1;
                const int jstride = this->info().jstride();
                const int kstride = this->info().kstride();

                const real *__restrict__ src = this->m_src->data();
                const real *__restrict__ coeff = this->m_coeff->data();
                real *__restrict__ dst = this->m_dst->data();

                assert(m_cache_info.istride() == 1);
                constexpr int icachestride = 1;
                const int jcachestride = m_cache_info.jstride();

#pragma omp parallel
                {
                    const int thread = omp_get_thread_num();
                    real *__restrict__ lap = m_lap_caches[thread].data() + m_cache_info.zero_offset();
                    real *__restrict__ flx = m_flx_caches[thread].data() + m_cache_info.zero_offset();
                    real *__restrict__ fly = m_fly_caches[thread].data() + m_cache_info.zero_offset();

#pragma omp for collapse(3)
                    for (int k = 0; k < ksize; ++k) {
                        for (int jb = 0; jb < jsize; jb += m_jblocksize) {
                            for (int ib = 0; ib < isize; ib += m_iblocksize) {
                                const int imax = std::min(ib + m_iblocksize, isize);
                                const int jmax = std::min(jb + m_jblocksize, jsize);

                                int index = (ib - 2) * istride + (jb - 2) * jstride + k * kstride;
                                int cacheindex = -2 * icachestride - 2 * jcachestride;
                                for (int j = jb - 2; j < jmax + 2; ++j) {
#pragma omp simd
                                    for (int i = ib - 2; i < imax + 2; ++i) {
                                        lap[cacheindex] =
                                            4 * src[index] - (src[index - istride] + src[index + istride] +
                                                                 src[index - jstride] + src[index + jstride]);

                                        index += istride;
                                        cacheindex += icachestride;
                                    }
                                    index += jstride - (imax - ib + 4) * istride;
                                    cacheindex += jcachestride - (imax - ib + 4) * icachestride;
                                }

                                index = (ib - 1) * istride + (jb - 1) * jstride + k * kstride;
                                cacheindex = -1 * icachestride - 1 * jcachestride;
                                for (int j = jb - 1; j < jmax + 1; ++j) {
#pragma omp simd
                                    for (int i = ib - 1; i < imax + 1; ++i) {
                                        flx[cacheindex] = lap[cacheindex + icachestride] - lap[cacheindex];
                                        if (flx[cacheindex] * (src[index + istride] - src[index]) > 0)
                                            flx[cacheindex] = 0;
                                        fly[cacheindex] = lap[cacheindex + jcachestride] - lap[cacheindex];
                                        if (fly[cacheindex] * (src[index + jstride] - src[index]) > 0)
                                            fly[cacheindex] = 0;
                                        index += istride;
                                        cacheindex += icachestride;
                                    }
                                    index += jstride - (imax - ib + 2) * istride;
                                    cacheindex += jcachestride - (imax - ib + 2) * icachestride;
                                }

                                index = ib * istride + jb * jstride + k * kstride;
                                cacheindex = 0;
                                for (int j = jb; j < jmax; ++j) {
#pragma omp simd
                                    for (int i = ib; i < imax; ++i) {
                                        dst[index] =
                                            src[index] -
                                            coeff[index] * (flx[cacheindex] - flx[cacheindex - icachestride] +
                                                               fly[cacheindex] - fly[cacheindex - jcachestride]);

                                        index += istride;
                                        cacheindex += icachestride;
                                    }
                                    index += jstride - (imax - ib) * istride;
                                    cacheindex += jcachestride - (imax - ib) * icachestride;
                                }
                            }
                        }
                    }
                }
            }

          private:
            int m_iblocksize, m_jblocksize;
            field_info m_cache_info;
            std::vector<std::vector<real, Allocator>> m_lap_caches, m_flx_caches, m_fly_caches;
        };
    } // namespace openmp
} // namespace backend
