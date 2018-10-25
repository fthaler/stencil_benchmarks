#include "backend/cuda.h"
#include "platform/nvidia/allocator.h"
#include "real.h"
#include "stencil_factory.h"

namespace platform {
    namespace nvidia {

        void register_stencils(stencil_factory &factory) {
            using allocator = cuda_allocator<real>;
            const std::string platform = "nvidia";

            backend::cuda::register_stencils<allocator>(factory, platform);
        }
    } // namespace nvidia
} // namespace platform
