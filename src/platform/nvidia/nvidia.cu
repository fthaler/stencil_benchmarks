#include "backend/cuda.h"
#include "platform/nvidia/nvidia.h"
#include "real.h"
#include "stencil_factory.h"

namespace platform {
    namespace nvidia {

        void register_stencils(stencil_factory &factory) {
            const std::string platform = "nvidia";

            backend::cuda::register_stencils(factory, platform);
        }
    } // namespace nvidia
} // namespace platform
