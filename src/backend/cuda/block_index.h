#pragma once

namespace backend {
    namespace cuda {

        template <int iminus_halo,
            int iplus_halo = iminus_halo,
            int jminus_halo = iminus_halo,
            int jplus_halo = jminus_halo>
        struct block_index {
            static_assert(
                iminus_halo >= 0 && iplus_halo >= 0 && jminus_halo >= 0 && jplus_halo >= 0, "invalid block halo size");
            int i, j;
            int iblock, jblock;
            int iblocksize, jblocksize;

            __device__ block_index(int iblocksize, int jblocksize) : iblocksize(iblocksize), jblocksize(jblocksize) {
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
                i = blockIdx.x * iblocksize + iblock;
                j = blockIdx.y * jblocksize + jblock;
            }

            __device__ bool in_block_or_halo() const {
                return iblock >= -iminus_halo && iblock < iblocksize + iplus_halo && jblock >= -jminus_halo &&
                       jblock < jblocksize + jplus_halo;
            }

            __device__ bool in_block() const {
                return iblock >= 0 && iblock < iblocksize && jblock >= 0 && jblock < jblocksize;
            }

          private:
            __device__ static constexpr int block_halo_padding(int x) {
                return x == 1 ? 1 : 2 * block_halo_padding((x + 1) / 2);
            }
        };

    } // namespace cuda
} // namespace backend
