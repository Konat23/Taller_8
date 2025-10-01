from functools import wraps
import sys
import time


def timing(f):
    """Decorator for measuring the execution time of methods."""

    @wraps(f)
    def wrapper(*args, **kwargs):
        ts = time.time()
        result = f(*args, **kwargs)
        te = time.time()
        elapsed = te - ts
        print("%r took %f s\n" % (f.__name__, elapsed))
        sys.stdout.flush()
        wrapper.elapsed_time = elapsed  # Guarda el tiempo en un atributo
        return result

    wrapper.elapsed_time = None  # Inicializa el atributo
    return wrapper
