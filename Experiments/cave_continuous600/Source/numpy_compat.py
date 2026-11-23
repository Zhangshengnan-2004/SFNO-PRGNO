"""Local NumPy compatibility for V2's use of the unchanged legacy Pypher."""
import numpy as np
import Pypher


class _PypherNumpy:
    # np.alltrue was an alias of np.all before NumPy 2.0.
    alltrue = staticmethod(np.all)

    def __getattr__(self, name):
        return getattr(np, name)


def install():
    """Adapt only Pypher's NumPy reference, leaving NumPy and V1 files intact."""
    if not hasattr(np, 'alltrue'):
        Pypher.np = _PypherNumpy()
