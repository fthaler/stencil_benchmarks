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
import unittest

import numpy as np

from stencil_benchmarks.tools import array


class TestAllocArray(unittest.TestCase):
    def test_basic(self):
        data = array.alloc_array((3, 5, 7), "int32", (0, 2, 1))
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype("int32"))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)

    def test_alignment(self):
        data = array.alloc_array((3, 5, 7), "int32", (0, 2, 1), alignment=64)
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype("int32"))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        self.assertTrue(data.ctypes.data % 64 == 0)
        self.assertTrue(data.strides[2] % 64 == 0)
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)

    def test_index_to_align(self):
        data = array.alloc_array(
            (3, 5, 7), "int32", (0, 2, 1), alignment=64, index_to_align=(1, 1, 2)
        )
        self.assertEqual(data.shape, (3, 5, 7))
        self.assertEqual(data.dtype, np.dtype("int32"))
        self.assertTrue(data.strides[1] < data.strides[2] < data.strides[0])
        self.assertTrue(data[1:, 1:, 2:].ctypes.data % 64 == 0)
        self.assertTrue(data.strides[2] % 64 == 0)
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    data[i, j, k] = i + 2 * j + 6 * k
        for i in range(3):
            for j in range(5):
                for k in range(7):
                    self.assertEqual(data[i, j, k], i + 2 * j + 6 * k)
