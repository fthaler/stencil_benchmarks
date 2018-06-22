#include <algorithm>

#include "field_repository.h"

field_repository::field_repository(const field_info &field_info, std::size_t field_array_size)
    : m_field_info(field_info), m_field_array_size(field_array_size) {}

void field_repository::cycle() {
    for (auto &callback : m_cycle_callbacks)
        callback();
}
