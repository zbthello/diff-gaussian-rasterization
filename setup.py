#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
os.path.dirname(os.path.abspath(__file__))

# setup()函数中变量的含义
# name: 在python中，import该模块的名称
# sources: 源代码文件的名称
# laungage: 默认为c，可以改成c++
# include_dirs: 传递给gcc或者g++编译器，include的头文件目录
# library_dirs: 传递给gcc或者g++编译器，指定链接文件的目录
# libraries: 传递给gcc或者g++编译器，指定链接的文件
# extra_compile_args: 额外的gcc或者g++编译参数
# extra_links_args: 额外的传给g++或者gcc编译器的链接参数
# define_macros: 定义的宏
# undefine_macros: 取消宏
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
            "cuda_rasterizer/rasterizer_impl.cu",
            "cuda_rasterizer/forward.cu",
            "cuda_rasterizer/backward.cu",
            "rasterize_points.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": ["-I" + os.path.join(os.path.dirname(os.path.abspath(__file__)), "third_party/glm/")]})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
