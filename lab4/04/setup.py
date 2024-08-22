from setuptools import setup
from torch.utils import cpp_extension

setup(

    name='hsigmoid_extension.so',
    version='0.1',
    ext_modules=[
        cpp_extension.CppExtension(
            'hsigmoid_extension',  # 模块名称
            ['hsigmoid.cpp'],  # 源文件
        ),
    ],

    cmdclass={
        'build_ext': cpp_extension.BuildExtension
    }

)

print("generate .so PASS!")