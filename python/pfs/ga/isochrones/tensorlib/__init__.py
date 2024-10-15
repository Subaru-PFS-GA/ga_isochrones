from .config import TENSORLIB

if TENSORLIB == 'numpy':
    from ._numpy import *
elif TENSORLIB == 'tensorflow':
    from ._tensorflow import *
elif TENSORLIB == 'pytorch':
    from ._pytorch import *
