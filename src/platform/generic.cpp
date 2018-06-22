#include <memory>

#include "arguments.h"
#include "platform/generic.h"
#include "real.h"

#include "platform/openmp_helper.h"

namespace platform {
    namespace generic {

        void register_stencils(stencil_factory &factory) {
            using allocator = std::allocator<real>;
            const std::string platform = "generic";

            register_openmp_stencils<allocator>(factory, platform);
        }

    } // namespace generic

} // namespace platform
