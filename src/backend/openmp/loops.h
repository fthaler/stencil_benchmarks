#pragma once

namespace backend {
    namespace openmp {
        template <class F>
        inline void loop_1d(int first, int last, F f) {
#pragma omp parallel for simd
            for (int i = first; i <= last; ++i)
                f(i);
        }
    } // namespace openmp
} // namespace backend
