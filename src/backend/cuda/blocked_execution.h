#pragma once

#include "arguments.h"
#include "backend/cuda/block_index.h"

namespace backend {
    namespace cuda {

        template <int IMinusHalo, int IPlusHalo = IMinusHalo, int JMinusHalo = IMinusHalo, int JPlusHalo = JMinusHalo>
        class blocked_execution {
          public:
            static constexpr int iminus_halo = IMinusHalo;
            static constexpr int iplus_halo = IPlusHalo;
            static constexpr int jminus_halo = JMinusHalo;
            static constexpr int jplus_halo = JPlusHalo;

            static void register_arguments(arguments &args) {
                args.add({"i-blocksize", "block size in i-direction", "32"})
                    .add({"j-blocksize", "block size in j-direction", "8"})
                    .add({"k-blocksize", "block size in k-direction", "1"});
            }

            blocked_execution(const arguments_map &args)
                : m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")),
                  m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");
                if (m_kblocksize <= 0)
                    throw ERROR("invalid k-blocksize");
            }

            runtime_parameters<block_index<iminus_halo, iplus_halo, jminus_halo, jplus_halo>> kernel_parameters(
                int isize, int jsize, int ksize) const {
                return {isize, jsize, ksize, m_iblocksize, m_jblocksize, m_kblocksize};
            }

          private:
            int m_iblocksize, m_jblocksize, m_kblocksize;
        };
    } // namespace cuda
} // namespace backend
