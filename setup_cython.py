"""Build script for the Cython extensions.

Usage:
    pip install cython                              # one-time
    python setup_cython.py build_ext --inplace      # compile

Extensions built:
    trickster.games.snapszer._fast_minimax  — Snapszer alpha-beta + PIMC (isolated snapszer/)
    ultisolver._solver_core                 — Ulti alpha-beta endgame solver (added in Stage 2)

After building, verify with:
    PYTHONPATH=snapszer python -c "from trickster.games.snapszer._fast_minimax import c_alphabeta; print('OK')"
"""

import os

from setuptools import setup, Extension
from Cython.Build import cythonize

_DIRECTIVES = {
    "boundscheck": False, "wraparound": False, "cdivision": True,
    "nonecheck": False, "language_level": "3",
}

extensions = [
    Extension(
        name="trickster.games.snapszer._fast_minimax",
        sources=["snapszer/trickster/games/snapszer/_fast_minimax.pyx"],
    ),
]
package_dir = {"": "snapszer"}
packages = ["trickster", "trickster.games.snapszer"]

# The decoupled Ulti solver core (Stage 2+). Guarded so the snapszer build works before it lands.
if os.path.exists("ultisolver/_solver_core.pyx"):
    extensions.append(Extension(name="ultisolver._solver_core",
                                sources=["ultisolver/_solver_core.pyx"]))

setup(
    name="trickster-cython-ext",
    ext_modules=cythonize(extensions, compiler_directives=_DIRECTIVES),
    package_dir=package_dir,
    packages=packages,
)
