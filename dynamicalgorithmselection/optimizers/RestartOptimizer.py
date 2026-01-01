from typing import Type

from dynamicalgorithmselection.optimizers.Optimizer import Optimizer


def restart_optimizer(base: Type[Optimizer]):
    class RestartOptimizer(base):
        def set_data(self, x=None, y=None, best_x=None, best_y=None, *args, **kwargs):
            pass

    new_name = f"{base.__name__}Restart"
    RestartOptimizer.__name__ = new_name
    RestartOptimizer.__qualname__ = new_name
    return RestartOptimizer
