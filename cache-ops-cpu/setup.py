from setuptools import setup, Extension
from torch.utils import cpp_extension

includes = ["./", "/home/ycy/repo/vllm/csrc"] + cpp_extension.include_paths()
sources = ["activation.cpp",
           "attention.cpp",
           "cache.cpp",
           "layernorm.cpp",
           "pos_encoding.cpp",
           "pybind.cpp"]


module = cpp_extension.CppExtension(
    name="vllm_cpu_ops",
    sources=sources,
    include_dirs=includes,
    extra_compile_args=['-g', '-O2', '-fopenmp', '-mavx512f', '-mavx512vl', '-mavx512bw', '-mavx512dq'],
)

setup(
    name="vllm_cpu_ops",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)