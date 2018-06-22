#include "stencil_factory.h"

stencil_factory::stencil_ptr stencil_factory::create(const arguments_map &args) {
    auto p = m_map.find(args.get("platform"));
    if (p == m_map.end())
        return nullptr;
    auto b = p->second.find(args.get("backend"));
    if (b == p->second.end())
        return nullptr;
    auto s = b->second.find(args.get("stencil"));
    if (s == b->second.end())
        return nullptr;
    return s->second(args);
}

std::ostream &operator<<(std::ostream &out, const stencil_factory &factory) {
    for (auto &p : factory.m_map) {
        out << "platform '" << p.first << "'\n";
        for (auto &b : p.second) {
            out << "  backend '" << b.first << "'\n";
            for (auto &s : b.second)
                out << "    " << s.first << "\n";
            out << "\n";
        }
        out << "\n";
    }
    return out;
}
