#pragma once

#include "backend/cuda/allocator.h"
#include "backend/cuda/blocked_execution.h"
#include "real.h"
#include "stencil/hdiff.h"

namespace backend {
    namespace cuda {

        class hdiff_otf : public stencil::hdiff<allocator<real>> {
          public:
            static void register_arguments(arguments &args);

            hdiff_otf(const arguments_map &args);

            void run() override;

          private:
            blocked_execution<0> m_blocked_execution;
        };

    } // namespace cuda
} // namespace backend
