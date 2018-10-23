#include "platform/generic.h"
#include "arguments.h"
#include "backend/openmp.h"
#include "real.h"
#include <memory>

namespace platform {
    namespace generic {

        void register_stencils(stencil_factory &factory) {
            using allocator = std::allocator<real>;
            const std::string platform = "generic";

            backend::openmp::register_stencils<allocator>(factory, platform);
        }

    } // namespace generic

} // namespace platform
