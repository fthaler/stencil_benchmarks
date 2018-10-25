#pragma once

#include <cmath>

using real = FLOAT_TYPE;

inline bool real_equal(float a, float b) {
    if (a == b) {
        return true;
    } else {
        float diff = std::abs(a - b);
        return diff / (a + b) < 1e-5f || diff < 1e-5f;
    }
}

inline bool real_equal(double a, double b) {
    if (a == b) {
        return true;
    } else {
        double diff = std::abs(a - b);
        return diff / (a + b) < 1e-12 || diff < 1e-12;
    }
}
