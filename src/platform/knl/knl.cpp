#include "platform/knl/knl.h"
#include "backend/openmp.h"
#include "platform/knl/allocator.h"
#include "real.h"
#include "stencil_factory.h"
#include <memory>

namespace platform {
    namespace knl {

        void register_stencils(stencil_factory &factory) {
            using allocator = knl::allocator<real>;
            const std::string platform = "knl";

            backend::openmp::register_stencils<allocator>(factory, platform);
        }

    } // namespace knl

} // namespace platform
