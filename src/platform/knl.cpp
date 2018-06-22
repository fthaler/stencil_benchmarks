#include <memory>

#include "arguments.h"
#include "platform/generic.h"
#include "platform/knl_allocator.h"
#include "real.h"

#include "platform/openmp_helper.h"

namespace platform {
    namespace knl {

        void register_stencils(stencil_factory &factory) {
            using allocator = knl_allocator<real>;
            const std::string platform = "knl";

            register_openmp_stencils<allocator>(factory, platform);
        }

    } // namespace knl

} // namespace platform
