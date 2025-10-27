## Dynamic Algorithm Selection

* In this Project I am performing initial research in constructing efficient RL-based DAS meta black box optimizers.

* It consists of 5 Metaheuristics: LM-CMAES, PSO, IPSO, PSOL and G3PCX.

* In order to simplify transition between them, all algorithms share the same population size (`n_individuals`).

* Main part of this project is a Reinforcement-Learning based agent, that can be trained to select correct metaheuristic during optimization.

* Agent chooses to switch algorithm (or not) in strictly quantized stages of optimization - after certain fraction of maximum function evaluations (`sub_optimization_ratio`) was used for a metaheuristic before.

* Script `main.py` starts training of the agent, after which trained RL model is evaluated and compared to its possible sub-optimizers (each alone).

### TO BE DONE

* Add command-line interface to specify sub-optimizers

* Add more sub-optimizers (OPOA methods etc.)

* Enable larger portfolio (size 4 and 5).


Much of the code is borrowed from https://github.com/Evolutionary-Intelligence/pypop/tree/main.