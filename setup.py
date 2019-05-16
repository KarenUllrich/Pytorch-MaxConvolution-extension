from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
from os.path import join

project_root = 'MaxConvolution_Module'
sources = [join(project_root, file) for file in ['max_convolution2d.cpp',
                                                 'max_convolution2d_sampler.cpp',
                                                 'max_convolution2d_cuda_kernel.cu']]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='max_convolution2d',
    version="0.1.0",
    author="Karen Ullrich",
    author_email="karn.ullrich@gmail.com",
    description="Max Convolution  module for pytorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/KarenUllrich/Pytorch-MaxConvolution-extension",
    install_requires=['torch>=1.0.1','numpy'],
    ext_modules=[
        CUDAExtension('max_convolution2d_sampler_backend',
                      sources,
                      extra_compile_args={'cxx': ['-fopenmp'], 'nvcc':[]},
                      extra_link_args=['-lgomp'])
    ],
    package_dir={'': project_root},
    packages=['max_convolution2d'],
    cmdclass={
        'build_ext': BuildExtension
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ])
