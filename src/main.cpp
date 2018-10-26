#include "arguments.h"
#include "except.h"
#include "platform/platform.h"
#include "stencil_execution.h"
#include "stencil_factory.h"
#include "table.h"
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

int main(int argc, char **argv) {
    arguments args(argv[0], "platform");

    args.add({"output", "output file", "stdout"});

    stencil_factory factory(args);

    platform::register_stencils(factory);

    auto argsmap = args.parse(argc, argv);

    auto stencil = factory.create(argsmap);

    double seconds = stencil->benchmark();
    double gbytes = stencil->touched_bytes() / 1.0e9;
    double bandwidth = gbytes / seconds;

    std::streambuf *buf;
    std::ofstream outfile;
    if (argsmap.get("output") == "stdout") {
        buf = std::cout.rdbuf();
    } else {
        outfile.open(argsmap.get("output"));
        if (!outfile)
            throw ERROR("could not open file '" + argsmap.get("output") + "'");
        buf = outfile.rdbuf();
    }
    std::ostream out(buf);

    table t(4);
    t << "stencil"
      << "executions"
      << "time"
      << "bandwidth";
    t << argsmap.get("stencil") << argsmap.get<int>("runs") << seconds << bandwidth;
    out << t;

    if (!stencil->verify()) {
        out << "WARNING: stencil verification failed!" << std::endl;
        return 1;
    }

    return 0;
}
