#pragma once

#include "backend/cuda/allocator.h"
#include "backend/cuda/blocked_execution.h"
#include "real.h"
#include "stencil/hdiff.h"

namespace backend {
    namespace cuda {

        class hdiff_otf_fillcache : public stencil::hdiff<allocator<real>> {
          public:
            static void register_arguments(arguments &args);

            hdiff_otf_fillcache(const arguments_map &args);

            void run() override;

          private:
            static constexpr int block_halo = 2;

            blocked_execution<block_halo> m_blocked_execution;
        };

    } // namespace cuda
} // namespace backend
