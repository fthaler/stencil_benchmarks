#pragma once

#include "backend/cuda/allocator.h"
#include "backend/cuda/blocked_execution.h"
#include "real.h"
#include "stencil/vadv.h"

namespace backend {
    namespace cuda {

        class vadv_kcache : public stencil::vadv<allocator<real>> {
          public:
            using blocked_execution_t = blocked_execution_2d<0>;

            static void register_arguments(arguments &args);

            vadv_kcache(const arguments_map &args);

            void run() override;

          private:
            blocked_execution_t m_blocked_execution;
        };

    } // namespace cuda
} // namespace backend
