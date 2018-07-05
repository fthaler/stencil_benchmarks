#pragma once

#include "backend/openmp.h"
#include "stencil_factory.h"

namespace platform {
    template <class Allocator>
    void register_openmp_stencils(stencil_factory &factory, const std::string &platform) {
        using namespace backend::openmp::stencil;

        auto name = [](const std::string &classname) {
            std::string cn = classname;
            std::replace(cn.begin(), cn.end(), '_', '-');
            return cn;
        };

#define REGISTER_STENCIL(stencil, args) \
    factory.register_stencil<stencil<Allocator>>(platform, "openmp", name(#stencil), args)

        REGISTER_STENCIL(basic_copy_1d, {});
        REGISTER_STENCIL(basic_avgi_1d, {});
        REGISTER_STENCIL(basic_avgj_1d, {});
        REGISTER_STENCIL(basic_avgk_1d, {});
        REGISTER_STENCIL(basic_lapij_1d, {});
        REGISTER_STENCIL(basic_lapik_1d, {});
        REGISTER_STENCIL(basic_lapjk_1d, {});
        REGISTER_STENCIL(basic_lapijk_1d, {});

        std::vector<std::tuple<std::string, std::string, std::string>> stencilargs = {
            std::make_tuple("i-blocksize", "block size in i-direction", "1024"),
            std::make_tuple("j-blocksize", "block size in j-direction", "1"),
            std::make_tuple("k-blocksize", "block size in k-direction", "1")};
        REGISTER_STENCIL(basic_copy_blocked, stencilargs);
        REGISTER_STENCIL(basic_avgi_blocked, stencilargs);
        REGISTER_STENCIL(basic_avgj_blocked, stencilargs);
        REGISTER_STENCIL(basic_avgk_blocked, stencilargs);
        REGISTER_STENCIL(basic_lapij_blocked, stencilargs);
        REGISTER_STENCIL(basic_lapik_blocked, stencilargs);
        REGISTER_STENCIL(basic_lapjk_blocked, stencilargs);
        REGISTER_STENCIL(basic_lapijk_blocked, stencilargs);

#undef REGISTER_STENCIL
    }
} // namespace platform
