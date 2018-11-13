#pragma once

#include "arguments.h"
#include "except.h"

namespace backend {
    namespace openmp {

        class blocked_execution_2d {
          public:
            struct diminfo {
                int size, blocksize, stride;
            };

            struct blockinfo {
                diminfo inner, outer;
            };

            static void register_arguments(arguments &args) {
                args.add({"i-blocksize", "block size in i-direction", "32"})
                    .add({"j-blocksize", "block size in j-direction", "32"});
            }

            blocked_execution_2d(const arguments_map &args)
                : m_isize(args.get<int>("i-size")), m_jsize(args.get<int>("j-size")),
                  m_iblocksize(args.get<int>("i-blocksize")), m_jblocksize(args.get<int>("j-blocksize")) {
                if (m_iblocksize <= 0)
                    throw ERROR("invalid i-blocksize");
                if (m_jblocksize <= 0)
                    throw ERROR("invalid j-blocksize");
            }

            blockinfo block(int istride, int jstride) const {
                blockinfo b{{m_isize, m_iblocksize, istride}, {m_jsize, m_jblocksize, jstride}};
                if (b.outer.stride < b.inner.stride)
                    std::swap(b.outer, b.inner);
                return b;
            }

          protected:
            int m_isize, m_jsize;
            int m_iblocksize, m_jblocksize;
        };

        class blocked_execution : blocked_execution_2d {
          public:
            struct blockinfo {
                diminfo inner, middle, outer;
            };

            static void register_arguments(arguments &args) {
                blocked_execution_2d::register_arguments(args);
                args.add({"k-blocksize", "block size in k-direction", "1"});
            }

            blocked_execution(const arguments_map &args)
                : blocked_execution_2d(args), m_ksize(args.get<int>("k-size")),
                  m_kblocksize(args.get<int>("k-blocksize")) {
                if (m_kblocksize <= 0)
                    throw ERROR("invalid k-blocksize");
            }

            blockinfo block(int istride, int jstride, int kstride) const {
                auto b2d = blocked_execution_2d::block(istride, jstride);
                blockinfo b{b2d.inner, b2d.outer, {m_ksize, m_kblocksize, kstride}};
                if (b.outer.stride < b.middle.stride)
                    std::swap(b.outer, b.middle);
                if (b.middle.stride < b.inner.stride)
                    std::swap(b.middle, b.inner);
                return b;
            }

          private:
            int m_ksize;
            int m_kblocksize;
        };

    } // namespace openmp
} // namespace backend
