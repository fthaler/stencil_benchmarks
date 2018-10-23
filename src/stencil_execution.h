#pragma once

#include <functional>

#include "arguments.h"
#include "field_repository.h"

class stencil_execution {
  public:
    static void register_arguments(arguments &args);

    stencil_execution(const arguments_map &args);
    virtual ~stencil_execution();

    virtual void run() = 0;
    virtual bool verify() = 0;

    virtual void prerun() {}
    virtual void postrun() {}

    double benchmark();
    virtual std::size_t touched_bytes() const = 0;

    const field_info &info() const;

  protected:
    template <class T, class Allocator>
    field_ptr<T, Allocator> create_field() {
        return m_repository.create_field<T, Allocator>();
    }

    void loop(std::function<void(int, int, int)> f, int halo = 0) const;
    bool loop_check(std::function<bool(int, int, int)> f, int halo = 0) const;

  private:
    field_repository m_repository;
    std::size_t m_runs;
};
