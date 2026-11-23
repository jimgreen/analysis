"""支持可选跨平台 C++17/OpenMP 原生微内核（``secore._native``）的打包脚本。

- Windows（MSVC）：使用 ``/O2 /std:c++17 /openmp``（通用 x86_64 指令集，兼容 Intel / AMD）；
- Linux / macOS（GCC / Clang）：使用 ``-O3 -std=c++17 -fopenmp``（兼容 x86_64 / ARM64）；
- 扩展标记为 ``optional=True``，无编译器的目标环境仍可顺利完成安装并自动回退纯 Python 实现。
"""

from __future__ import annotations

import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class _OptionalBuildExt(build_ext):
    """按编译器自动注入通用优化与 OpenMP 选项；编译失败时优雅回退纯 Python。"""

    def build_extensions(self):
        ctype = self.compiler.compiler_type
        for ext in self.extensions:
            if ctype == "msvc":
                ext.extra_compile_args = ["/O2", "/std:c++17", "/openmp"]
            elif sys.platform == "darwin":
                ext.extra_compile_args = ["-O3", "-std=c++17"]
            else:
                ext.extra_compile_args = ["-O3", "-std=c++17", "-fopenmp"]
                ext.extra_link_args = ["-fopenmp"]
        super().build_extensions()


ext_modules = [
    Extension(
        name="secore._native",
        sources=["secore/_native.cpp"],
        language="c++",
        optional=True,
    )
]

setup(
    ext_modules=ext_modules,
    cmdclass={"build_ext": _OptionalBuildExt},
)
