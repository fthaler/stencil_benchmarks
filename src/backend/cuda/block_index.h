#pragma once

#include "except.h"
#include <cassert>

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        struct block_index {
            static constexpr int iminus_halo = IMinusHalo;
            static constexpr int iplus_halo = IPlusHalo;
            static constexpr int jminus_halo = JMinusHalo;
            static constexpr int jplus_halo = JPlusHalo;

            static_assert(
                iminus_halo >= 0 && iplus_halo >= 0 && jminus_halo >= 0 && jplus_halo >= 0, "invalid block halo size");
            int iblocksize, jblocksize, kblocksize;
            int i, j, k;
            int iblock, jblock, kblock;

            __device__ block_index(int isize, int jsize, int ksize)
                : iblocksize(blockDim.x), jblocksize(blockDim.y - jminus_halo - jplus_halo - (iminus_halo > 0 ? 1 : 0) -
                                                     (iplus_halo > 0 ? 1 : 0)),
                  kblocksize(blockDim.z) {
                compute_block_indices();
                const int iblockbase = blockIdx.x * iblocksize;
                const int jblockbase = blockIdx.y * jblocksize;
                const int kblockbase = blockIdx.z * kblocksize;
                i = iblockbase + iblock;
                j = jblockbase + jblock;
                k = kblockbase + kblock;
                if (iblockbase + iblocksize > isize)
                    iblocksize = isize - iblockbase;
                if (jblockbase + jblocksize > jsize)
                    jblocksize = jsize - jblockbase;
                if (kblockbase + kblocksize > ksize)
                    kblocksize = ksize - kblockbase;
                assert(iblocksize > 0);
                assert(jblocksize > 0);
                assert(kblocksize > 0);
            }

            __device__ bool in_block_or_halo() const {
                return iblock >= -iminus_halo && iblock < iblocksize + iplus_halo && jblock >= -jminus_halo &&
                       jblock < jblocksize + jplus_halo && kblock >= 0 && kblock < kblocksize;
            }

            __device__ bool in_block() const {
                return iblock >= 0 && iblock < iblocksize && jblock >= 0 && jblock < jblocksize && kblock >= 0 &&
                       kblock < kblocksize;
            }

          private:
            template <bool AllZero = (iminus_halo == 0 && iplus_halo == 0 && jminus_halo == 0 && jplus_halo == 0)>
            __device__ typename std::enable_if<AllZero>::type compute_block_indices() {
                iblock = threadIdx.x;
                jblock = threadIdx.y;
                kblock = threadIdx.z;
            }

            template <bool AllZero = (iminus_halo == 0 && iplus_halo == 0 && jminus_halo == 0 && jplus_halo == 0)>
            __device__ typename std::enable_if<!AllZero>::type compute_block_indices() {
                const int jboundary_limit = jblocksize + jminus_halo + jplus_halo;
                const int iminus_limit = jboundary_limit + (iminus_halo > 0 ? 1 : 0);
                const int iplus_limit = iminus_limit + (iplus_halo > 0 ? 1 : 0);

                iblock = -iminus_halo - 1;
                jblock = -jminus_halo - 1;
                if (threadIdx.y < jboundary_limit) {
                    iblock = threadIdx.x;
                    jblock = threadIdx.y - jminus_halo;
                } else if (threadIdx.y < iminus_limit) {
                    constexpr int pad = block_halo_padding(iminus_halo);
                    iblock = -pad + int(threadIdx.x) % pad;
                    jblock = int(threadIdx.x) / pad - jminus_halo;
                } else if (threadIdx.y < iplus_limit) {
                    constexpr int pad = block_halo_padding(iplus_halo);
                    iblock = int(threadIdx.x) % pad + iblocksize;
                    jblock = int(threadIdx.x) / pad - jminus_halo;
                }
                kblock = threadIdx.z;
            }

            __device__ static constexpr int block_halo_padding(int x) {
                return x == 0 ? 0 : x == 1 ? 1 : 2 * block_halo_padding((x + 1) / 2);
            }
        };

        template <class BlockIndex>
        struct runtime_parameters {
            runtime_parameters(int isize, int jsize, int ksize, int iblocksize, int jblocksize, int kblocksize)
                : m_isize(isize), m_jsize(jsize), m_ksize(ksize), m_iblocksize(iblocksize), m_jblocksize(jblocksize),
                  m_kblocksize(kblocksize) {
                constexpr int max_i_halo =
                    BlockIndex::iminus_halo > BlockIndex::iplus_halo ? BlockIndex::iminus_halo : BlockIndex::iplus_halo;
                if ((jblocksize + BlockIndex::jminus_halo + BlockIndex::jplus_halo) * max_i_halo >
                    std::min(iblocksize, 32)) {
                    throw ERROR("unsupported block size");
                }
            }

            dim3 blocks() const {
                return dim3(rndup_div(m_isize, m_iblocksize),
                    rndup_div(m_jsize, m_jblocksize),
                    rndup_div(m_ksize, m_kblocksize));
            }

            dim3 threads() const {
                return dim3(m_iblocksize,
                    m_jblocksize + BlockIndex::jminus_halo + BlockIndex::jplus_halo +
                        (BlockIndex::iminus_halo > 0 ? 1 : 0) + (BlockIndex::iplus_halo > 0 ? 1 : 0),
                    m_kblocksize);
            }

            int blocksize_with_halo() const {
                return (m_iblocksize + BlockIndex::iminus_halo + BlockIndex::iplus_halo) *
                       (m_jblocksize + BlockIndex::jminus_halo + BlockIndex::jplus_halo) * m_kblocksize;
            }

          private:
            static constexpr int rndup_div(int den, int div) { return (den + div - 1) / div; }

            int m_isize, m_jsize, m_ksize;
            int m_iblocksize, m_jblocksize, m_kblocksize;
        };

    } // namespace cuda
} // namespace backend
