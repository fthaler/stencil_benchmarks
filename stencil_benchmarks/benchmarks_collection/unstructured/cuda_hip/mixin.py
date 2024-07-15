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
import abc
import contextlib
import ctypes
import warnings
from pathlib import Path

import numpy as np

from stencil_benchmarks.benchmark import (
    Benchmark,
    ExecutionError,
    Parameter,
    ParameterError,
)
from stencil_benchmarks.tools import array, compilation, cpphelpers, template
from stencil_benchmarks.benchmarks_collection.stencils.cuda_hip import api


class UnstructuredMixin(Benchmark):
    compiler = Parameter("compiler path", "nvcc")
    compiler_flags = Parameter("compiler flags", "")
    backend = Parameter("use NVIDIA CUDA or AMD HIP", "cuda", choices=["cuda", "hip"])
    gpu_architecture = Parameter("GPU architecture", dtype=str, nargs=1)
    print_code = Parameter("print generated code", False)
    dry_runs = Parameter("kernel dry-runs before the measurement", 0)
    timers = Parameter("timer type", default="gpu", choices=["gpu", "wall", "hip-ext"])

    def setup(self):
        super().setup()

        if self.backend == "cuda" and self.timers == "hip-ext":
            raise ParameterError("hip-ext timers are not compatible with CUDA")

        template_file = (
            Path(__file__).parent.resolve() / "templates" / self.template_file()
        )
        code = template.render(template_file, **self.template_args())
        code = cpphelpers.format_code(code, line_numbers=False)

        if self.print_code:
            print(cpphelpers.format_code(code, line_numbers=True))

        try:
            self.compiled = compilation.GnuLibrary(
                code, [self.compiler] + self.compiler_flags.split(),
                extension=".cu" if self.backend == "cuda" else ".cpp",
            )
        except compilation.CompilationError as error:
            raise ParameterError(*error.args) from error

        if self.verify and self.dry_runs:
            warnings.warn(
                "using --dry-runs together with verification might lead to "
                "false negatives for stencils with read-write fields"
            )

    def default_compiler_flags(self):
        flags = "-std=c++11 -DNDEBUG"
        if self.backend == "cuda":
            if self.compiler.endswith("nvcc"):
                flags += " -x cu -Xcompiler -Wall"
                if self.gpu_architecture:
                    flags += " -arch " + self.gpu_architecture
            elif self.compiler.endswith("clang++"):
                flags += " -xcuda -Ofast -Wall -lcudart"
                if self.gpu_architecture:
                    flags += " --cuda-gpu-arch=" + self.gpu_architecture
        elif self.backend == "hip":
            flags += " -xhip -Ofast -Wall"
            if self.gpu_architecture:
                flags += " --amdgpu-target=" + self.gpu_architecture
        return flags

    @abc.abstractmethod
    def template_file(self):
        pass

    def template_args(self):
        return dict(
            args=[name for name, _, _ in self.args],
            backend=self.backend,
            ctype=compilation.dtype_cname(self.dtype),
            nbtype=compilation.dtype_cname(self.neighbor_table_dtype),
            strides={
                name: self.strides(data)
                for (name, _, _), data in zip(self.args, self._data[0])
            }
            | {
                "v2e": self.strides(self._v2e_table),
                "e2v": self.strides(self._e2v_table),
            },
            nproma=self.nproma,
            nvertices=self.nvertices,
            nedges=self.nedges,
            nlevels=self.domain[2],
            v2e_max_neighbors=self._v2e_table.shape[1],
            e2v_max_neighbors=self._e2v_table.shape[1],
            skip_values=self.skip_values,
            alignment=self.alignment,
            dry_runs=self.dry_runs,
            timers=self.timers,
        )

    @contextlib.contextmanager
    def on_device(self, data):
        runtime = api.runtime(self.backend)

        device_data = [
            array.alloc_array(
                d.shape,
                d.dtype,
                np.argsort(d.strides)[::-1],
                self.alignment,
                alloc=runtime.malloc,
                apply_offset=self.offset_allocations,
            )
            for d in data
        ]

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                device_array.ctypes.data,
                host_array.ctypes.data,
                array.nbytes(host_array),
                "HostToDevice",
            )
        runtime.device_synchronize()

        yield device_data

        for host_array, device_array in zip(data, device_data):
            runtime.memcpy(
                host_array.ctypes.data,
                device_array.ctypes.data,
                array.nbytes(host_array),
                "DeviceToHost",
            )
        runtime.device_synchronize()

    def run_stencil(self, data):
        data = [self._v2e_table, self._e2v_table] + list(data)
        with self.on_device(data) as device_data:
            data_ptrs = [
                compilation.data_ptr(device_array)
                for device_array in device_data
            ]

            time = ctypes.c_double()
            try:
                self.compiled.kernel(ctypes.byref(time), *data_ptrs)
            except compilation.ExecutionError as error:
                raise ExecutionError() from error

        return dict(time=time.value)
