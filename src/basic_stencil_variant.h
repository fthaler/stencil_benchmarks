#pragma once

#include <cmath>
#include <limits>
#include <random>

#include "except.h"
#include "variant_base.h"

namespace platform {

    template <class Platform, class ValueType>
    class basic_stencil_variant : public variant_base {
      public:
        using platform = Platform;
        using value_type = ValueType;
        using allocator = typename platform::template allocator<value_type>;

        basic_stencil_variant(const arguments_map &args);
        virtual ~basic_stencil_variant() {}

        std::vector<std::string> stencil_list() const override;

        virtual void copy() = 0;
        virtual void copyi() = 0;
        virtual void copyj() = 0;
        virtual void copyk() = 0;
        virtual void avgi() = 0;
        virtual void avgj() = 0;
        virtual void avgk() = 0;
        virtual void sumi() = 0;
        virtual void sumj() = 0;
        virtual void sumk() = 0;
        virtual void lapij() = 0;

      protected:
        value_type *src() { return m_src_data.data() + zero_offset(); }
        value_type *dst() { return m_dst_data.data() + zero_offset(); }

        std::function<void()> stencil_function(const std::string &stencil) override;

        bool verify(const std::string &stencil) override;

        std::size_t touched_elements(const std::string &stencil) const override;
        std::size_t bytes_per_element() const override { return sizeof(value_type); }

      private:
        std::vector<value_type, allocator> m_src_data, m_dst_data;
        value_type *m_src, *m_dst;
    };

    template <class Platform, class ValueType>
    basic_stencil_variant<Platform, ValueType>::basic_stencil_variant(const arguments_map &args)
        : variant_base(args), m_src_data(storage_size()), m_dst_data(storage_size()) {
#pragma omp parallel
        {
            std::minstd_rand eng;
            std::uniform_real_distribution<value_type> dist(-100, 100);

            int total_size = storage_size();
#pragma omp for
            for (int i = 0; i < total_size; ++i) {
                m_src_data.at(i) = dist(eng);
                m_dst_data.at(i) = dist(eng);
            }
        }
    }

    template <class Platform, class ValueType>
    std::vector<std::string> basic_stencil_variant<Platform, ValueType>::stencil_list() const {
        return {"copy", "copyi", "copyj", "copyk", "avgi", "avgj", "avgk", "sumi", "sumj", "sumk", "lapij"};
    }

    template <class Platform, class ValueType>
    std::function<void()> basic_stencil_variant<Platform, ValueType>::stencil_function(const std::string &stencil) {
        if (stencil == "copy")
            return std::bind(&basic_stencil_variant::copy, this);
        if (stencil == "copyi")
            return std::bind(&basic_stencil_variant::copyi, this);
        if (stencil == "copyj")
            return std::bind(&basic_stencil_variant::copyj, this);
        if (stencil == "copyk")
            return std::bind(&basic_stencil_variant::copyk, this);
        if (stencil == "avgi")
            return std::bind(&basic_stencil_variant::avgi, this);
        if (stencil == "avgj")
            return std::bind(&basic_stencil_variant::avgj, this);
        if (stencil == "avgk")
            return std::bind(&basic_stencil_variant::avgk, this);
        if (stencil == "sumi")
            return std::bind(&basic_stencil_variant::sumi, this);
        if (stencil == "sumj")
            return std::bind(&basic_stencil_variant::sumj, this);
        if (stencil == "sumk")
            return std::bind(&basic_stencil_variant::sumk, this);
        if (stencil == "lapij")
            return std::bind(&basic_stencil_variant::lapij, this);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

    template <class Platform, class ValueType>
    bool basic_stencil_variant<Platform, ValueType>::verify(const std::string &stencil) {
        std::function<bool(int, int, int)> f;
        auto s = [&](int i, int j, int k) { return (m_src_data.data() + zero_offset())[index(i, j, k)]; };
        auto d = [&](int i, int j, int k) { return (m_dst_data.data() + zero_offset())[index(i, j, k)]; };

        if (stencil == "copy") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k); };
        } else if (stencil == "copyi") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i + 1, j, k); };
        } else if (stencil == "copyj") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j + 1, k); };
        } else if (stencil == "copyk") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k + 1); };
        } else if (stencil == "avgi") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i - 1, j, k) + s(i + 1, j, k); };
        } else if (stencil == "avgj") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j - 1, k) + s(i, j + 1, k); };
        } else if (stencil == "avgk") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k - 1) + s(i, j, k + 1); };
        } else if (stencil == "sumi") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k) + s(i + 1, j, k); };
        } else if (stencil == "sumj") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k) + s(i, j + 1, k); };
        } else if (stencil == "sumk") {
            f = [&](int i, int j, int k) { return d(i, j, k) == s(i, j, k) + s(i, j, k + 1); };
        } else if (stencil == "lapij") {
            f = [&](int i, int j, int k) {
                return d(i, j, k) == s(i, j, k) + s(i - 1, j, k) + s(i + 1, j, k) + s(i, j - 1, k) + s(i, j + 1, k);
            };
        } else {
            throw ERROR("unknown stencil '" + stencil + "'");
        }

        const int isize = this->isize();
        const int jsize = this->jsize();
        const int ksize = this->ksize();
        bool success = true;
#pragma omp parallel for collapse(3) reduction(&& : success)
        for (int k = 0; k < ksize; ++k)
            for (int j = 0; j < jsize; ++j)
                for (int i = 0; i < isize; ++i)
                    success = success && f(i, j, k);
        return success;
    }

    template <class Platform, class ValueType>
    std::size_t basic_stencil_variant<Platform, ValueType>::touched_elements(const std::string &stencil) const {
        std::size_t i = isize();
        std::size_t j = jsize();
        std::size_t k = ksize();
        if (stencil == "copy" || stencil == "copyi" || stencil == "copyj" || stencil == "copyk")
            return i * j * k * 2;
        if (stencil == "avgi")
            return i * j * k + (i + 2) * j * k;
        if (stencil == "avgj")
            return i * j * k + i * (j + 2) * k;
        if (stencil == "avgk")
            return i * j * k + i * j * (k + 2);
        if (stencil == "sumi")
            return i * j * k + (i + 1) * j * k;
        if (stencil == "sumj")
            return i * j * k + i * (j + 1) * k;
        if (stencil == "sumk")
            return i * j * k + i * j * (k + 1);
        if (stencil == "lapij")
            return i * j * k + (i + 2) * (j + 2) * (k + 2);
        throw ERROR("unknown stencil '" + stencil + "'");
    }

} // platform
