#pragma once

#include "field_info.h"
#include "real.h"

#ifdef __CUDACC__
#define FUNCTOR_ATTR __host__ __device__ __forceinline__
#else
#define FUNCTOR_ATTR [[gnu::always_inline]]
#endif

namespace stencil {

    struct functor_base {
        functor_base(const field_info &info, const real *src, real *dst)
            : src(src), dst(dst), istride(info.istride()), jstride(info.jstride()), kstride(info.kstride()) {}

        const real *__restrict__ src;
        real *__restrict__ dst;
        const int istride, jstride, kstride;
    };

    struct copy_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index]; }
    };

    struct copyi_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index + istride]; }
    };

    struct copyj_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index + jstride]; }
    };

    struct copyk_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index + kstride]; }
    };

    struct avgi_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index - istride] + src[index + istride]; }
    };

    struct avgj_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index - jstride] + src[index + jstride]; }
    };

    struct avgk_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const { dst[index] = src[index - kstride] + src[index + kstride]; }
    };

    struct lapij_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const {
            dst[index] =
                src[index] + src[index - istride] + src[index + istride] + src[index - jstride] + src[index + jstride];
        }
    };

    struct lapik_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const {
            dst[index] =
                src[index] + src[index - istride] + src[index + istride] + src[index - kstride] + src[index + kstride];
        }
    };

    struct lapjk_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const {
            dst[index] =
                src[index] + src[index - jstride] + src[index + jstride] + src[index - kstride] + src[index + kstride];
        }
    };

    struct lapijk_functor : functor_base {
        using functor_base::functor_base;

        FUNCTOR_ATTR void operator()(int index) const {
            dst[index] = src[index] + src[index - istride] + src[index + istride] + src[index - jstride] +
                         src[index + jstride] + src[index - kstride] + src[index + kstride];
        }
    };

} // namespace stencil

#undef FUNCTOR_ATTR
