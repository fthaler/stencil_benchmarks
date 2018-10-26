#pragma once

#include "backend/cuda/allocator.h"
#include "real.h"
#include "stencil/hdiff.h"

namespace backend {
    namespace cuda {

        class hdiff_naive : public stencil::hdiff<allocator<real>> {
          public:
            static void register_arguments(arguments &args);

            hdiff_naive(const arguments_map &args);

            void run() override;

          private:
            field_ptr<real, allocator<real>> m_lap, m_flx, m_fly;
            int m_iblocks, m_jblocks, m_kblocks;
            int m_ithreads, m_jthreads, m_kthreads;
        };

    } // namespace cuda
} // namespace backend
