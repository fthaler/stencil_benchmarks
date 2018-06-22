#include <functional>
#include <vector>

#include "field_info.h"

template <class T, class Allocator = std::allocator<T>>
class field_array {
  public:
    field_array(const field_info &info, std::size_t size);
    virtual ~field_array() {}

    const T *data() const { return m_data[m_current].data() + m_field_info.zero_offset(); }
    T *data() { return m_data[m_current].data() + m_field_info.zero_offset(); }

    const T &operator()(int i, int j, int k) const { return data()[m_field_info.index(i, j, k)]; }
    T &operator()(int i, int j, int k) { return data()[m_field_info.index(i, j, k)]; }

    void cycle();

    void fill(std::function<T(int, int, int)> f);

  private:
    std::vector<std::vector<T, Allocator>> m_data;
    field_info m_field_info;
    std::size_t m_current;
};

template <class T, class Allocator>
field_array<T, Allocator>::field_array(const field_info &info, std::size_t size) : m_field_info(info), m_current(0) {
    for (std::size_t i = 0; i < size; ++i)
        m_data.emplace_back(info.storage_size());
}

template <class T, class Allocator>
void field_array<T, Allocator>::cycle() {
    m_current = (m_current + 1) % m_data.size();
}

template <class T, class Allocator>
void field_array<T, Allocator>::fill(std::function<T(int, int, int)> f) {
    const int halo = m_field_info.halo();
    const int isize = m_field_info.isize();
    const int jsize = m_field_info.jsize();
    const int ksize = m_field_info.ksize();
    for (auto &d : m_data) {
        T *ptr = d.data() + m_field_info.zero_offset();
#pragma omp parallel for collapse(3)
        for (int k = -halo; k < ksize + halo; ++k) {
            for (int j = -halo; j < jsize + halo; ++j) {
                for (int i = -halo; i < isize + halo; ++i) {
                    ptr[m_field_info.index(i, j, k)] = f(i, j, k);
                }
            }
        }
    }
}
