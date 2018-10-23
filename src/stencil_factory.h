#pragma once

#include <iostream>
#include <map>
#include <string>

#include "arguments.h"
#include "stencil_execution.h"

class stencil_factory {
  public:
    using stencil_ptr = std::unique_ptr<stencil_execution>;
    using stencil_creator = std::function<stencil_ptr(const arguments_map &)>;

    stencil_factory(arguments &args) : m_args(args) {}

    template <class Stencil>
    void register_stencil(const std::string &platform, const std::string &backend, const std::string &name) {
        m_map[platform][backend][name] = [](const arguments_map &args) { return stencil_ptr(new Stencil(args)); };

        auto &sc = m_args.command(platform, "backend").command(backend, "stencil").command(name);
        Stencil::register_arguments(sc);
    }

    stencil_ptr create(const arguments_map &args);

  private:
    friend std::ostream &operator<<(std::ostream &, const stencil_factory &);

    std::map<std::string, std::map<std::string, std::map<std::string, stencil_creator>>> m_map;
    arguments &m_args;
};

std::ostream &operator<<(std::ostream &out, const stencil_factory &factory);
