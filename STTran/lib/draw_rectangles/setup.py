from distutils.core import setup
from distutils.extension import Extension
import numpy

try:
    from Cython.Build import cythonize
    ext_modules = cythonize("draw_rectangles.pyx")
except Exception:
    # Fallback for offline clusters without Cython preinstalled.
    ext_modules = [Extension("draw_rectangles", ["draw_rectangles.c"])]

setup(
    name="draw_rectangles_cython",
    ext_modules=ext_modules,
    include_dirs=[numpy.get_include()],
)