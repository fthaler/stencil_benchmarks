#pragma once

#include "config.h"
#include "stencil_factory.h"

#ifdef SBENCH_WITH_PLATFORM_GENERIC
#include "generic/generic.h"
#endif
#ifdef SBENCH_WITH_PLATFORM_KNL
#include "knl/knl.h"
#endif
#ifdef SBENCH_WITH_PLATFORM_NVIDIA
#include "nvidia/nvidia.h"
#endif

namespace platform {
    inline void register_stencils(stencil_factory &factory) {
#ifdef SBENCH_WITH_PLATFORM_GENERIC
        generic::register_stencils(factory);
#endif
#ifdef SBENCH_WITH_PLATFORM_KNL
        knl::register_stencils(factory);
#endif
#ifdef SBENCH_WITH_PLATFORM_NVIDIA
        nvidia::register_stencils(factory);
#endif
    }
} // namespace platform
