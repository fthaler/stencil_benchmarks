#pragma once

#include "backend/util.h"
#include "stencil_factory.h"
#ifdef __CUDACC__
#include "backend/cuda/basic_stencils_1d.h"
#include "backend/cuda/basic_stencils_blocked.h"
#endif

namespace backend {
    namespace cuda {

        template <class Allocator>
        void register_stencils(stencil_factory &factory, const std::string &platform) {
#ifdef __CUDACC__
            using namespace stencil;
#define REGISTER_STENCIL(stencil) \
    factory.register_stencil<stencil<Allocator>>(platform, "cuda", underscore_to_dash(#stencil));
#else
#define REGISTER_STENCIL(stencil)
#endif

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
    } // namespace cuda
} // namespace backend
