import pkgutil
import importlib
import inspect

available_optimizers = {}


def collect_classes(package_name):
    package = importlib.import_module(package_name)

    for _, module_name, is_pkg in pkgutil.iter_modules(
        package.__path__, package.__name__ + "."
    ):
        if is_pkg:
            collect_classes(module_name)
        else:
            module = importlib.import_module(module_name)
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if (
                    obj.__module__ == module.__name__
                    and not obj.__name__ == "Optimizer"
                    and not package_name.endswith(name)
                ):
                    available_optimizers[name] = obj


# Collect all classes starting from this package
collect_classes(__name__)
