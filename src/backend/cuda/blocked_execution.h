#pragma once

#include "arguments.h"
#include "backend/cuda/block_index.h"

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        class blocked_execution {
          public:
            using block_index_t = block_index<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo>;

            static constexpr int iminus_halo() { return IMinusHalo; }
            static constexpr int iplus_halo() { return IPlusHalo; }
            static constexpr int jminus_halo() { return JMinusHalo; }
            static constexpr int jplus_halo() { return JPlusHalo; }
            template <bool UniformHalo = (IMinusHalo == IPlusHalo && IMinusHalo == JMinusHalo &&
                                          IMinusHalo == JPlusHalo)>
            static constexpr typename std::enable_if<UniformHalo, int>::type halo() {
                return IMinusHalo;
            }

            static void register_arguments(arguments &args) {
                args.add({"i-blocksize", "block size in i-direction", "32"})
                    .add({"j-blocksize", "block size in j-direction", "8"})
                    .add({"k-blocksize", "block size in k-direction", "1"});
            }

            blocked_execution(const arguments_map &args)
                : m_isize(args.get<int>("i-size")), m_jsize(args.get<int>("j-size")), m_ksize(args.get<int>("k-size")),
                  m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")),
                  m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");
                if (m_kblocksize <= 0)
                    throw ERROR("invalid k-blocksize");
                constexpr int max_i_halo = IMinusHalo > IPlusHalo ? IMinusHalo : IPlusHalo;
                if ((m_jblocksize + JMinusHalo + JPlusHalo) * max_i_halo > std::min(m_iblocksize, 32)) {
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
                    m_jblocksize + JMinusHalo + JPlusHalo + (IMinusHalo > 0 ? 1 : 0) + (IPlusHalo > 0 ? 1 : 0),
                    m_kblocksize);
            }

            int blocksize_with_halo() const {
                return (m_iblocksize + IMinusHalo + IPlusHalo) * (m_jblocksize + JMinusHalo + JPlusHalo) * m_kblocksize;
            }

          private:
            static constexpr int rndup_div(int den, int div) { return (den + div - 1) / div; }

            int m_isize, m_jsize, m_ksize;
            int m_iblocksize, m_jblocksize, m_kblocksize;
        };
    } // namespace cuda
} // namespace backend
