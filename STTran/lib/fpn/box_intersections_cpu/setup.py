from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Build import cythonize
    ext_modules = cythonize("bbox.pyx")
except Exception:
    # Fallback for offline clusters without Cython preinstalled.
    ext_modules = [Extension("bbox", ["bbox.c"])]

setup(
    name="bbox_cython",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)