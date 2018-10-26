#include "util.h"
#include <algorithm>

std::string underscore_to_dash(const std::string &s) {
    std::string sd = s;
    std::replace(sd.begin(), sd.end(), '_', '-');
    return sd;
}
