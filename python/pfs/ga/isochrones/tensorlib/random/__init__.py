from ..config import TENSORLIB

if TENSORLIB == 'numpy':
    from ._random_numpy import *
elif TENSORLIB == 'tensorflow':
    from ._random_tensorflow import *
elif TENSORLIB == 'pytorch':
    from ._random_pytorch import *
