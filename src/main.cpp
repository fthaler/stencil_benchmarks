#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>

#include "arguments.h"
#include "except.h"
#include "stencil_execution.h"
#include "stencil_factory.h"
#include "table.h"

#include "platform/generic.h"
#include "platform/knl.h"

int main(int argc, char **argv) {
    arguments args(argv[0], "platform");

    args.add("output", "output file", "stdout");

    stencil_execution::register_arguments(args);

    stencil_factory factory(args);

    platform::generic::register_stencils(factory);
    platform::knl::register_stencils(factory);

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

    return 0;
}
