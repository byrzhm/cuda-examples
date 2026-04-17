from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='my_kernel',
    ext_modules=[
        CUDAExtension(
            name='my_kernel',
            sources=['kernel.cu', 'binding.cpp'],
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
