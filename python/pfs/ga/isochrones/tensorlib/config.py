import importlib

if importlib.util.find_spec('torch') is not None:
    TENSORLIB = 'pytorch'
elif importlib.util.find_spec('tensorflow') is not None:
    TENSORLIB = 'tensorflow'
else:
    TENSORLIB = 'numpy'