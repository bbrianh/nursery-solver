import pkgutil
import importlib
import inspect

__all__ = []
SOLVERS = {}

for _, module_name, _ in pkgutil.iter_modules(__path__):
    module = importlib.import_module(f"{__name__}.{module_name}")

    # find classes inside module
    for name, cls in inspect.getmembers(module, inspect.isclass):
        # only include classes defined in that file
        if cls.__module__ == module.__name__:
            
            # OPTIONAL: filter only Solver-like classes
            if name == "Solver" or name.endswith("Solver"):
                
                # store in registry
                SOLVERS[name] = cls

                # expose to "from solvers import *"
                globals()[name] = cls
                __all__.append(name)