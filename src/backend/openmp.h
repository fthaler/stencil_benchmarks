#pragma once

#include "backend/openmp/basic_stencils_1d.h"
#include "backend/openmp/basic_stencils_blocked.h"
#include "backend/util.h"

namespace backend {
    namespace openmp {

        template <class Allocator>
        void register_stencils(stencil_factory &factory, const std::string& platform) {
            using namespace stencil;
#define REGISTER_STENCIL(stencil) \
    factory.register_stencil<stencil<Allocator>>(platform, "openmp", underscore_to_dash(#stencil));

        REGISTER_STENCIL(basic_copy_1d);
        REGISTER_STENCIL(basic_avgi_1d);
        REGISTER_STENCIL(basic_avgj_1d);
        REGISTER_STENCIL(basic_avgk_1d);
        REGISTER_STENCIL(basic_lapij_1d);
        REGISTER_STENCIL(basic_lapik_1d);
        REGISTER_STENCIL(basic_lapjk_1d);
        REGISTER_STENCIL(basic_lapijk_1d);

        REGISTER_STENCIL(basic_copy_blocked);
        REGISTER_STENCIL(basic_avgi_blocked);
        REGISTER_STENCIL(basic_avgj_blocked);
        REGISTER_STENCIL(basic_avgk_blocked);
        REGISTER_STENCIL(basic_lapij_blocked);
        REGISTER_STENCIL(basic_lapik_blocked);
        REGISTER_STENCIL(basic_lapjk_blocked);
        REGISTER_STENCIL(basic_lapijk_blocked);

#undef REGISTER_STENCIL
        }
    }
}
