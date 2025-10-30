import pkgutil
import importlib
import inspect

available_optimizers = {}

# Iterate over all modules in this package
for loader, module_name, is_pkg in pkgutil.iter_modules(__path__):
    if not is_pkg:  # skip sub-packages for now
        module = importlib.import_module(f".{module_name}", package=__name__)

        # Inspect members and collect classes defined in the module
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if obj.__module__ == module.__name__ and name != "Optimizer":
                available_optimizers[name] = obj
                globals()[name] = obj  # optional: expose at package level
