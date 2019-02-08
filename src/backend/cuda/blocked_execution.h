#pragma once

#include "arguments.h"
#include "backend/cuda/block_index.h"

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        class blocked_execution_2d {
          public:
            using block_index_t = block_index_2d<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo>;

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
                    .add({"j-blocksize", "block size in j-direction", "8"});
            }

            blocked_execution_2d(const arguments_map &args)
                : m_isize(args.get<int>("i-size")), m_jsize(args.get<int>("j-size")),
                  m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");
                constexpr int max_i_halo = IMinusHalo > IPlusHalo ? IMinusHalo : IPlusHalo;
                if ((m_jblocksize + JMinusHalo + JPlusHalo) * max_i_halo > std::min(m_iblocksize, 32)) {
                    throw ERROR("unsupported block size");
                }
            }

            dim3 blocks() const { return dim3(rndup_div(m_isize, m_iblocksize), rndup_div(m_jsize, m_jblocksize), 1); }

            dim3 threads() const {
                return dim3(m_iblocksize,
                    m_jblocksize + JMinusHalo + JPlusHalo + (IMinusHalo > 0 ? 1 : 0) + (IPlusHalo > 0 ? 1 : 0),
                    1);
            }

            int blocksize_with_halo() const {
                return (m_iblocksize + IMinusHalo + IPlusHalo) * (m_jblocksize + JMinusHalo + JPlusHalo);
            }

          protected:
            static constexpr int rndup_div(int den, int div) { return (den + div - 1) / div; }

            int m_isize, m_jsize;
            int m_iblocksize, m_jblocksize;
        };

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        class blocked_execution : blocked_execution_2d<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo> {
            using base_t = blocked_execution_2d<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo>;

          public:
            using block_index_t = block_index<IMinusHalo, IPlusHalo, JMinusHalo, JPlusHalo>;

            static void register_arguments(arguments &args) {
                base_t::register_arguments(args);
                args.add({"k-blocksize", "block size in k-direction", "1"});
            }

            blocked_execution(const arguments_map &args)
                : base_t(args), m_ksize(args.get<int>("k-size")), m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_kblocksize <= 0)
                    throw ERROR("invalid k-blocksize");
            }

            dim3 blocks() const {
                dim3 blocks_2d = base_t::blocks();
                return dim3(blocks_2d.x, blocks_2d.y, base_t::rndup_div(m_ksize, m_kblocksize));
            }

            dim3 threads() const {
                dim3 threads_2d = base_t::threads();
                return dim3(threads_2d.x, threads_2d.y, m_kblocksize);
            }

            int blocksize_with_halo() const { return base_t::blocksize_with_halo() * m_kblocksize; }

          protected:
            int m_ksize;
            int m_kblocksize;
        };
    } // namespace cuda
} // namespace backend
