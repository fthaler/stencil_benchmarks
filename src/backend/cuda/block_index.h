#pragma once

#include "except.h"
#include <cassert>

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        struct block_index {
            __device__ static constexpr int iminus_halo() { return IMinusHalo; }
            __device__ static constexpr int iplus_halo() { return IPlusHalo; }
            __device__ static constexpr int jminus_halo() { return JMinusHalo; }
            __device__ static constexpr int jplus_halo() { return JPlusHalo; }
            template <bool UniformHalo = (IMinusHalo == IPlusHalo && IMinusHalo == JMinusHalo &&
                                          IMinusHalo == JPlusHalo)>
            __device__ static constexpr typename std::enable_if<UniformHalo, int>::type halo() {
                return IMinusHalo;
            }

            static_assert(
                IMinusHalo >= 0 && IPlusHalo >= 0 && JMinusHalo >= 0 && JPlusHalo >= 0, "invalid block halo size");
            int iblocksize, jblocksize, kblocksize;
            int i, j, k;
            int iblock, jblock, kblock;

            __device__ block_index(int isize, int jsize, int ksize)
                : iblocksize(blockDim.x),
                  jblocksize(blockDim.y - JMinusHalo - JPlusHalo - (IMinusHalo > 0 ? 1 : 0) - (IPlusHalo > 0 ? 1 : 0)),
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
                return iblock >= -IMinusHalo && iblock < iblocksize + IPlusHalo && jblock >= -JMinusHalo &&
                       jblock < jblocksize + JPlusHalo && kblock >= 0 && kblock < kblocksize;
            }

            __device__ bool in_block() const {
                return iblock >= 0 && iblock < iblocksize && jblock >= 0 && jblock < jblocksize && kblock >= 0 &&
                       kblock < kblocksize;
            }

          private:
            template <bool AllZero = (IMinusHalo == 0 && IPlusHalo == 0 && JMinusHalo == 0 && JPlusHalo == 0)>
            __device__ typename std::enable_if<AllZero>::type compute_block_indices() {
                iblock = threadIdx.x;
                jblock = threadIdx.y;
                kblock = threadIdx.z;
            }

            template <bool AllZero = (IMinusHalo == 0 && IPlusHalo == 0 && JMinusHalo == 0 && JPlusHalo == 0)>
            __device__ typename std::enable_if<!AllZero>::type compute_block_indices() {
                const int jboundary_limit = jblocksize + JMinusHalo + JPlusHalo;
                const int iminus_limit = jboundary_limit + (IMinusHalo > 0 ? 1 : 0);
                const int iplus_limit = iminus_limit + (IPlusHalo > 0 ? 1 : 0);

                iblock = -IMinusHalo - 1;
                jblock = -JMinusHalo - 1;
                if (threadIdx.y < jboundary_limit) {
                    iblock = threadIdx.x;
                    jblock = threadIdx.y - JMinusHalo;
                } else if (threadIdx.y < iminus_limit) {
                    constexpr int pad = block_halo_padding(IMinusHalo);
                    iblock = -pad + int(threadIdx.x) % pad;
                    jblock = int(threadIdx.x) / pad - JMinusHalo;
                } else if (threadIdx.y < iplus_limit) {
                    constexpr int pad = block_halo_padding(IPlusHalo);
                    iblock = int(threadIdx.x) % pad + iblocksize;
                    jblock = int(threadIdx.x) / pad - JMinusHalo;
                }
                kblock = threadIdx.z;
            }

            __device__ static constexpr int block_halo_padding(int x) {
                return x == 0 ? 0 : x == 1 ? 1 : 2 * block_halo_padding((x + 1) / 2);
            }
        };

    } // namespace cuda
} // namespace backend
