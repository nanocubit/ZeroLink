"""
ZeroLink v2.0
Enterprise-Ready Zero-Copy IPC Runtime for PyTorch
"""

import sys
from setuptools import setup, find_packages


def parse_requirements(filename: str) -> list[str]:
    """
    Считывает зависимости из requirements-файла.
    Игнорирует комментарии, пустые строки и include-директивы (-r ...).
    """
    requirements = []
    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#') or line.startswith('-r '):
                    continue
                requirements.append(line)
    except FileNotFoundError:
        return []
    return requirements


# Парсинг базовых зависимостей
install_requires = parse_requirements('requirements.txt')

extras_require = {
    "gpu": [
        "torch>=1.12.0",
        "cuda-python>=11.7",
    ],
    "ray": [
        "ray>=2.0.0",
    ],
    "cgpu": [
        "cgpu>=0.1.0",
    ],
    "dev": [
        "pytest>=7.0",
        "pytest-cov",
        "black",
        "mypy",
    ],
}
extras_require["full"] = sorted({dep for group in extras_require.values() for dep in group})

# Список расширений
ext_modules = []
cmdclass = {}

# Проверяем флаг --no-cuda
# Если он есть, мы НЕ компилируем C++ модуль (полезно для чисто-Python окружений или CI)
if "--no-cuda" in sys.argv:
    sys.argv.remove("--no-cuda")
else:
    try:
        from torch.utils.cpp_extension import BuildExtension, CUDAExtension

        ext_modules.append(
            CUDAExtension(
                name='zerolink.core.gpu.ext.ipc_ext',
                sources=['zerolink/core/gpu/ext/ipc_ext.cpp'],
                extra_link_args=['-lcuda'],
            )
        )
        cmdclass = {'build_ext': BuildExtension}
    except ImportError:
        # Разрешаем установку core-профиля без torch/cuda toolchain.
        ext_modules = []
        cmdclass = {}

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
    cmdclass=cmdclass,
    install_requires=install_requires,
    extras_require=extras_require,
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
