"""
ZeroLink v2.0
Enterprise-Ready Zero-Copy IPC Runtime for PyTorch
"""

import sys
from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

def parse_requirements(filename: str) -> list[str]:
    """
    Считывает зависимости из файла requirements.txt.
    Игнорирует комментарии и пустые строки.
    """
    requirements = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#'):
                    requirements.append(line)
    except FileNotFoundError:
        return []

# Парсинг зависимостей
install_requires = parse_requirements('requirements.txt')

# Список расширений
ext_modules = []

# Проверяем флаг --no-cuda
# Если он есть, мы НЕ компилируем C++ модуль (полезно для чисто-Python окружений или CI)
if "--no-cuda" not in sys.argv:
    ext_modules.append(
        CUDAExtension(
            name='zerolink.core.gpu.ext.ipc_ext',
            sources=['zerolink/core/gpu/ext/ipc_ext.cpp'],
            extra_link_args=['-lcuda'],
            # Опционально: можно добавить define_macros для детальной настройки
            # define_macros=[('TORCH_EXTENSION_NAME', 'zerolink.core.gpu.ext.ipc_ext')],
        )
    )
else:
    # Удаляем флаг, чтобы setuptools не ругался на лишний аргумент
    if "--no-cuda" in sys.argv:
        sys.argv.remove("--no-cuda")

setup(
    name="zerolink",
    version="2.0.0",
    author="ZeroLink Team",
    description="High-Performance Zero-Copy IPC Runtime for PyTorch with Enterprise-grade reliability",
    long_description=open('README.md', 'r', encoding='utf-8').read(),
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/zerolink",

    packages=find_packages(exclude=['tests*', 'docs*', 'scripts*', 'deployment*']),
    ext_modules=ext_modules,

    cmdclass={'build_ext': BuildExtension},

    install_requires=install_requires,
    python_requires='>=3.8',

    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)