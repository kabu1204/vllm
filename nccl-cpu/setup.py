import os
import torch
from setuptools import setup
from torch.utils import cpp_extension

sources = ["nccl-cpu.cpp", "cuda_ipc.cpp"]
include_dirs = [f"{os.path.dirname(os.path.abspath(__file__))}/"]

if torch.cuda.is_available():
    module = cpp_extension.CUDAExtension(
        name="nccl_cpu",
        sources=sources,
        include_dirs=include_dirs,
    )
else:
    module = cpp_extension.CppExtension(
        name="nccl_cpu",
        sources=sources,
        include_dirs=include_dirs,
    )

setup(
    name="nccl-cpu",
    version="0.0.1",
    ext_modules=[module],
    cmdclass={'build_ext': cpp_extension.BuildExtension}
)