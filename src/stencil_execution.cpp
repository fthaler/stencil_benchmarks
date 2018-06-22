#include <chrono>

#include "stencil_execution.h"

stencil_execution::stencil_execution(const arguments_map &args)
    : m_repository(field_info(args.get<int>("i-size"),
                       args.get<int>("j-size"),
                       args.get<int>("k-size"),
                       args.get<int>("i-layout"),
                       args.get<int>("j-layout"),
                       args.get<int>("k-layout"),
                       args.get<int>("halo"),
                       args.get<int>("alignment")),
          1),
      m_runs(args.get<int>("runs")) {}

stencil_execution::~stencil_execution() {}

double stencil_execution::benchmark() {
    using clock = std::chrono::high_resolution_clock;
    std::size_t dry = m_repository.array_size();

    prerun();

    for (std::size_t s = 0; s < dry; ++s) {
        run();
        m_repository.cycle();
    }

    auto start = clock::now();

    for (std::size_t s = 0; s < m_runs; ++s) {
        run();
        m_repository.cycle();
    }

    auto end = clock::now();

    postrun();

    return std::chrono::duration<double>(end - start).count() / m_runs;
}

const field_info &stencil_execution::info() const { return m_repository.info(); }

void stencil_execution::register_arguments(arguments &args) {
    args.add("i-size", "domain size in i-direction", "1024")
        .add("j-size", "domain size in j-direction", "1024")
        .add("k-size", "domain size in k-direction", "80")
        .add("i-layout", "layout specifier", "2")
        .add("j-layout", "layout specifier", "1")
        .add("k-layout", "layout specifier", "0")
        .add("halo", "halo size", "2")
        .add("alignment", "alignment in elements", "1")
        .add("runs", "number of runs", "20");
}

void stencil_execution::loop(std::function<void(int, int, int)> f, int halo) const {
    const int isize = info().isize();
    const int jsize = info().jsize();
    const int ksize = info().ksize();
#pragma omp parallel for collapse(3)
    for (int k = -halo; k < ksize + halo; ++k)
        for (int j = -halo; j < jsize + halo; ++j)
            for (int i = -halo; i < isize + halo; ++i)
                f(i, j, k);
}

bool stencil_execution::loop_check(std::function<bool(int, int, int)> f, int halo) const {
    const int isize = info().isize();
    const int jsize = info().jsize();
    const int ksize = info().ksize();
    bool result = true;
#pragma omp parallel for collapse(3) reduction(&& : result)
    for (int k = -halo; k < ksize + halo; ++k)
        for (int j = -halo; j < jsize + halo; ++j)
            for (int i = -halo; i < isize + halo; ++i)
                result &= f(i, j, k);
    return result;
}
