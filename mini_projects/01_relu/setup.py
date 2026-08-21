from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="relu",
    ext_modules=[
        CUDAExtension(
            name="relu",
            sources=["src/relu_bindings.cpp","src/relu.cu"],
        )
    ],
    cmdclass={"build_ext":BuildExtension},
)