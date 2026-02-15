from typing import Type, TypeVar

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer

# Create a TypeVar that is bound to the Optimizer base class
T = TypeVar("T", bound=Optimizer)


def restart_optimizer(base: Type[T]) -> Type[T]:
    class RestartOptimizer(base):  # type: ignore[misc, valid-type]
        def set_data(self, x=None, y=None, best_x=None, best_y=None, *args, **kwargs):
            # We override this to do nothing, effectively "restarting"
            # or ignoring previous state transitions.
            pass

    new_name = f"{base.__name__}Restart"
    RestartOptimizer.__name__ = new_name
    RestartOptimizer.__qualname__ = new_name

    # Casting or returning as Type[T] ensures mypy sees the
    # result as the same category of class as the input.
    return RestartOptimizer
