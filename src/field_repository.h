#pragma once

#include <functional>
#include <memory>
#include <vector>

#include "field_array.h"
#include "field_info.h"

template <class T, class Allocator>
using field_ptr = std::shared_ptr<field_array<T, Allocator>>;

class field_repository {
  public:
    field_repository(const field_info &field_info, std::size_t field_array_size);

    template <class T, class Allocator>
    field_ptr<T, Allocator> create_field();

    void cycle();

    const field_info &info() const { return m_field_info; }
    const std::size_t &array_size() const { return m_field_array_size; }

  private:
    field_info m_field_info;
    std::size_t m_field_array_size;
    std::vector<std::function<void()>> m_cycle_callbacks;
};

template <class T, class Allocator>
field_ptr<T, Allocator> field_repository::create_field() {
    auto f = std::make_shared<field_array<T, Allocator>>(m_field_info, m_field_array_size);
    auto w = std::weak_ptr<field_array<T, Allocator>>(f);
    m_cycle_callbacks.push_back([w]() {
        if (auto s = w.lock())
            s->cycle();
    });
    return f;
}
