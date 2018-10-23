#include "stencil_factory.h"
#include "platform/nvidia/allocator.h"
#include "real.h"
#include "backend/cuda.h"

namespace platform {
    namespace nvidia {
    
        void register_stencils(stencil_factory &factory) {
            using allocator = cuda::allocator<real>;
            const std::string platform = "cuda";

            backend::cuda::register_stencils<allocator>(factory, platform);
        }
    }
} // namespace platform
