#pragma once

#include "backend/openmp/basic_stencils_1d.h"
#include "backend/openmp/basic_stencils_blocked.h"
#include "backend/openmp/hdiff_ijcache_blocked.h"
#include "backend/openmp/hdiff_naive.h"
#include "backend/openmp/hdiff_otf.h"
#include "backend/openmp/hdiff_otf_blocked.h"
#include "backend/openmp/vadv_kcache_blocked.h"
#include "stencil_factory.h"
#include "util.h"

namespace backend {
    namespace openmp {

        template <class Allocator>
        void register_stencils(stencil_factory &factory, const std::string &platform) {
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

            REGISTER_STENCIL(hdiff_ijcache_blocked);
            REGISTER_STENCIL(hdiff_naive);
            REGISTER_STENCIL(hdiff_otf);
            REGISTER_STENCIL(hdiff_otf_blocked);

            REGISTER_STENCIL(vadv_kcache_blocked);

#undef REGISTER_STENCIL
        }
    } // namespace openmp
} // namespace backend
