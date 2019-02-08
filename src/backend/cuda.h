#pragma once

#include "stencil_factory.h"

namespace backend {
    namespace cuda {

        void register_stencils(stencil_factory &factory, const std::string &platform);

    } // namespace cuda
} // namespace backend
