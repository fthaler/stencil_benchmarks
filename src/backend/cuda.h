#pragma once

#include "backend/cuda/basic_stencils_1d.h"
#include "backend/cuda/basic_stencils_blocked.h"
#include "backend/cuda/hdiff_naive.h"
#include "backend/cuda/hdiff_otf.h"
#include "stencil_factory.h"
#include "util.h"

namespace backend {
    namespace cuda {

        template <class Allocator>
        void register_stencils(stencil_factory &factory, const std::string &platform) {
            using namespace stencil;
#define REGISTER_STENCIL(stencil) \
    factory.register_stencil<stencil<Allocator>>(platform, "cuda", underscore_to_dash(#stencil));

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

            REGISTER_STENCIL(hdiff_naive);
            REGISTER_STENCIL(hdiff_otf);

#undef REGISTER_STENCIL
        }
    } // namespace cuda
} // namespace backend
