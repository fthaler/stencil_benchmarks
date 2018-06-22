#pragma once

#include "backend/openmp.h"
#include "stencil_factory.h"

namespace platform {
    template <class Allocator>
    void register_openmp_stencils(stencil_factory &factory, const std::string &platform) {
        using namespace backend::openmp::stencil;

        factory.register_stencil<basic_copy_1d<Allocator>>(platform, "openmp", "basic-copy-1d");
        factory.register_stencil<basic_avgi_1d<Allocator>>(platform, "openmp", "basic-avgi-1d");
        factory.register_stencil<basic_avgj_1d<Allocator>>(platform, "openmp", "basic-avgj-1d");
        factory.register_stencil<basic_avgk_1d<Allocator>>(platform, "openmp", "basic-avgk-1d");
        factory.register_stencil<basic_lapij_1d<Allocator>>(platform, "openmp", "basic-lapij-1d");
        factory.register_stencil<basic_lapik_1d<Allocator>>(platform, "openmp", "basic-lapik-1d");
        factory.register_stencil<basic_lapjk_1d<Allocator>>(platform, "openmp", "basic-lapjk-1d");
        factory.register_stencil<basic_lapijk_1d<Allocator>>(platform, "openmp", "basic-lapijk-1d");
    }
} // namespace platform
