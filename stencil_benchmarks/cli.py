# Stencil Benchmarks
#
# Copyright (c) 2017-2021, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its contributors
# may be used to endorse or promote products derived from this software without
# specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.
#
# SPDX-License-Identifier: BSD-3-Clause
import copy
import gc
import re
import sys
import types
import warnings

import click

from . import benchmark
from .tools import cli as cli_tools
from .tools import validation


def _cli_command(bmark):
    command_path = bmark.__module__.split(".")[2:]
    command_name = re.sub("(?!^)([A-Z1-9]+)", r"-\1", bmark.__name__).lower()
    return command_path + [command_name]


def _instantiate(bmark, skip_invalid_parameters, **kwargs):
    try:
        return bmark(**kwargs)
    except benchmark.ParameterError as error:
        if skip_invalid_parameters:
            return
        click.echo(*error.args)
        sys.exit(1)


def _run_instance(bmark_instance, skip_execution_failures):
    try:
        results = bmark_instance.run()
    except benchmark.ExecutionError as error:
        if skip_execution_failures:
            return
        click.echo("\nexecution error: " + " ".join(error.args).strip())
        sys.exit(1)
    except validation.ValidationError as error:
        click.echo("\nvalidation error: " + " ".join(error.args).strip())
        sys.exit(2)
    assert results

    if isinstance(results, dict):
        results = [results]

    result_keys = set(results[0].keys())
    pretty_params = cli_tools.pretty_parameters(bmark_instance)
    results = [dict(r, **pretty_params) for r in results]

    return results, result_keys


def _process_results(results, result_keys, output, report):
    if not results:
        click.echo("no data collected")
        sys.exit(1)

    import pandas as pd

    table = pd.DataFrame(results)
    if output:
        table.to_csv(output)

    is_num = pd.api.types.is_numeric_dtype
    result_keys = {k for k in result_keys if is_num(table.dtypes[k])}

    nunique = table.apply(pd.Series.nunique)
    if any(nunique > 1):
        table.drop(nunique[nunique <= 1].index, axis=1, inplace=True)

    if report == "full":
        click.echo(table.to_string())
    else:
        group_keys = list(set(table.columns) - result_keys)
        if group_keys:
            medians = table.groupby(group_keys).median()
            if report == "best-median":
                best = medians["time"].idxmin()
                click.echo(medians.loc[[best]])
            else:
                click.echo(medians.sort_values(by="time").to_string())
        else:
            click.echo(table.median().to_string())


def _bmark_options(bmark):
    for name, param in bmark.parameters.items():
        name = "--" + name.replace("_", "-")
        description = param.description[0].upper() + param.description[1:] + "."
        if param.dtype is bool:
            option = click.option(
                name + "/" + name.replace("--", "--no-"),
                default=param.default,
                help=description,
                show_default=True,
            )
        else:
            if param.dtype is str and param.choices:
                dtype = click.Choice(param.choices)
            else:
                dtype = param.dtype
            option = click.option(
                name,
                type=cli_tools.range_type(dtype),
                nargs=param.nargs,
                help=description,
                required=param.default is None,
                default=param.default,
                show_default=param.default is not None,
            )
        yield option


def _cli_func(bmark):
    @click.pass_context
    def func(ctx, **kwargs):
        unpacked_kwargs = list(cli_tools.unpack_ranges(**kwargs))

        with warnings.catch_warnings(record=True) as catched_warnings:
            results = []
            try:
                with cli_tools.ProgressBar() as progress:
                    for kws in progress.report(unpacked_kwargs):
                        bmark_instance = _instantiate(
                            bmark, ctx.obj.skip_invalid_parameters, **kws
                        )
                        if bmark_instance is None:
                            continue

                        for _ in progress.report(range(ctx.obj.executions)):
                            result = _run_instance(
                                bmark_instance, ctx.obj.skip_execution_failures
                            )
                            if result is None:
                                continue
                            instance_results, result_keys = result
                            results += instance_results

                        try:
                            del bmark_instance
                            gc.collect()
                        except NameError:
                            pass
            except KeyboardInterrupt:
                pass
            for warning in {str(w.message) for w in catched_warnings}:
                print("WARNING:", warning)

        _process_results(
            results, result_keys, output=ctx.obj.output, report=ctx.obj.report
        )

    for option in _bmark_options(bmark):
        func = option(func)
    return func


@click.group()
@click.option("--executions", "-e", type=int, default=3)
@click.option(
    "--report",
    "-r",
    default="all-medians",
    type=click.Choice(["best-median", "all-medians", "full"]),
)
@click.option("--skip-invalid-parameters", "-s", is_flag=True)
@click.option("--skip-execution-failures", "-q", is_flag=True)
@click.option("--output", "-o", type=click.Path())
@click.pass_context
def _cli(
    ctx, executions, report, skip_invalid_parameters, skip_execution_failures, output
):
    ctx.obj.executions = executions
    ctx.obj.report = report
    ctx.obj.skip_invalid_parameters = skip_invalid_parameters
    ctx.obj.skip_execution_failures = skip_execution_failures
    ctx.obj.output = output


def _build(commands):
    hierarchy = dict()

    for command, func in commands:
        current = hierarchy
        for subcommand in command[:-1]:
            current = current.setdefault(subcommand, dict())
        current[command[-1]] = func

    @click.pass_context
    def empty(_):
        pass

    def build_click_hierarchy(group, subcommands):
        for subcommand, subcommand_subcommands in subcommands.items():
            if isinstance(subcommand_subcommands, dict):
                build_click_hierarchy(
                    group.group(name=subcommand.replace("_", "-"))(empty),
                    subcommand_subcommands,
                )
            else:
                group.command(name=subcommand)(subcommand_subcommands)

    main_group = copy.copy(_cli)
    build_click_hierarchy(main_group, hierarchy)
    return main_group


def main(*args, **kwargs):
    commands = [(_cli_command(bmark), _cli_func(bmark)) for bmark in benchmark.REGISTRY]
    main_group = _build(commands)

    return main_group(*args, **kwargs, obj=types.SimpleNamespace())
