#pragma once

#include "except.h"
#include <cassert>

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        struct block_index_2d {
            static_assert(
                IMinusHalo >= 0 && IPlusHalo >= 0 && JMinusHalo >= 0 && JPlusHalo >= 0, "invalid block halo size");

            __device__ static constexpr int iminus_halo() { return IMinusHalo; }
            __device__ static constexpr int iplus_halo() { return IPlusHalo; }
            __device__ static constexpr int jminus_halo() { return JMinusHalo; }
            __device__ static constexpr int jplus_halo() { return JPlusHalo; }
            template <bool UniformHalo = (IMinusHalo == IPlusHalo && IMinusHalo == JMinusHalo &&
                                          IMinusHalo == JPlusHalo)>
            __device__ __forceinline__ static constexpr typename std::enable_if<UniformHalo, int>::type halo() {
                return IMinusHalo;
            }

            int iblocksize, jblocksize, kblocksize;
            int i, j, k;
            int iblock, jblock;

            __device__ __forceinline__ block_index_2d(int isize, int jsize)
                : iblocksize(blockDim.x),
                  jblocksize(blockDim.y - JMinusHalo - JPlusHalo - (IMinusHalo > 0 ? 1 : 0) - (IPlusHalo > 0 ? 1 : 0)) {
                compute_block_indices();
                const int iblockbase = blockIdx.x * iblocksize;
                const int jblockbase = blockIdx.y * jblocksize;
                i = iblockbase + iblock;
                j = jblockbase + jblock;
                if (iblockbase + iblocksize > isize)
                    iblocksize = isize - iblockbase;
                if (jblockbase + jblocksize > jsize)
                    jblocksize = jsize - jblockbase;
                assert(iblocksize > 0);
                assert(jblocksize > 0);
            }

            __device__ __forceinline__ bool in_block_or_halo(
                int iminus_halo, int iplus_halo, int jminus_halo, int jplus_halo) const {
                assert(iminus_halo <= IMinusHalo);
                assert(iplus_halo <= IPlusHalo);
                assert(jminus_halo <= JMinusHalo);
                assert(jplus_halo <= JPlusHalo);
                return iblock >= -iminus_halo && iblock < iblocksize + iplus_halo && jblock >= -jminus_halo &&
                       jblock < jblocksize + jplus_halo;
            }

            __device__ __forceinline__ bool in_block_or_halo() const {
                return in_block_or_halo(IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo);
            }

            __device__ __forceinline__ bool in_block() const {
                return iblock >= 0 && iblock < iblocksize && jblock >= 0 && jblock < jblocksize;
            }

            __device__ __forceinline__ int blocksize_with_halo() const {
                return (iblocksize + IMinusHalo + IPlusHalo) * (jblocksize + JMinusHalo + JPlusHalo);
            }

          private:
            template <bool AllZero = (IMinusHalo == 0 && IPlusHalo == 0 && JMinusHalo == 0 && JPlusHalo == 0)>
            __device__ __forceinline__ typename std::enable_if<AllZero>::type compute_block_indices() {
                iblock = threadIdx.x;
                jblock = threadIdx.y;
            }

            template <bool AllZero = (IMinusHalo == 0 && IPlusHalo == 0 && JMinusHalo == 0 && JPlusHalo == 0)>
            __device__ __forceinline__ typename std::enable_if<!AllZero>::type compute_block_indices() {
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
            }

            __device__ __forceinline__ static constexpr int block_halo_padding(int x) {
                return x == 0 ? 0 : x == 1 ? 1 : 2 * block_halo_padding((x + 1) / 2);
            }
        };

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        struct block_index : block_index_2d<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo> {
            using base_t = block_index_2d<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo>;

            int kblocksize;
            int k;
            int kblock;

            __device__ __forceinline__ block_index(int isize, int jsize, int ksize)
                : base_t(isize, jsize), kblocksize(blockDim.z), kblock(threadIdx.z) {
                const int kblockbase = blockIdx.z * kblocksize;
                k = kblockbase + kblock;
                if (kblockbase + kblocksize > ksize)
                    kblocksize = ksize - kblockbase;
                assert(kblocksize > 0);
            }

            __device__ __forceinline__ bool in_block_or_halo(
                int iminus_halo, int iplus_halo, int jminus_halo, int jplus_halo) const {
                return base_t::in_block_or_halo(iminus_halo, iplus_halo, jminus_halo, jplus_halo) &&
                       kblock < kblocksize;
            }

            __device__ __forceinline__ bool in_block_or_halo() const {
                return in_block_or_halo(IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo);
            }

            __device__ __forceinline__ bool in_block() const { return base_t::in_block() && kblock < kblocksize; }

            __device__ __forceinline__ int blocksize_with_halo() const {
                return base_t::blocksize_with_halo() * kblocksize;
            }
        };

    } // namespace cuda
} // namespace backend
