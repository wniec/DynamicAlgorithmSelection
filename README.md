# Dynamic Algorithm Selection (DAS)

This project explores **Reinforcement Learning (RL)-based meta–black-box optimization**, where an agent dynamically selects among several metaheuristics to optimize performance.

---

## Overview

- Implements **five metaheuristic algorithms**:  
  **LM-CMAES**, **PSO**, **IPSO**, **PSOL**, and **G3PCX**
- All algorithms share the same **population size** (`n_individuals`) for seamless transitions.
- The core of the system is an **RL-based agent** that learns to **switch** between optimizers during the optimization process.
- The agent makes switching decisions in **quantized stages**, determined by the fraction of total function evaluations (`sub_optimization_ratio`) already consumed by the current optimizer.
- The entry script, **`main.py`**, launches training of the RL agent, followed by evaluation and comparison with individual sub-optimizers.

---

## Features & Structure

| Component                 | Description                                        |
|---------------------------|----------------------------------------------------|
| **RL Agent**              | Learns dynamic optimizer selection policies        |
| **Metaheuristics**        | LM-CMAES, PSO, IPSO, PSOL, G3PCX                   |
| **Population Handling**   | Shared `n_individuals` across all optimizers       |
| **Evaluation**            | Trained RL model compared to standalone optimizers |

---

## Command-Line Interface (CLI)

The project supports flexible configuration through command-line arguments.

### **Usage**
```bash
python3 main.py <name> [options]
```


| Argument                           | Type               | Default                     | Description                                        |
|------------------------------------|--------------------|-----------------------------|----------------------------------------------------|
| `name`                             | `str` (positional) | —                           | **Required name tag** for the run or experiment    |
| `-p`, `--portfolio`                | `list[str]`        | `['SPSO', 'IPSO', 'SPSOL']` | Portfolio of sub-optimizers to include             |
| `-m`, `--population_size`          | `int`              | `20`                        | Population size for all fixed-pop-size optimizers  |
| `-f`, `--fe_multiplier`            | `int`              | `10_000`                    | Function evaluation multiplier                     |
| `-s`, `--sub_optimization_ratio`   | `int`              | `10`                        | ratio of max_fe, for each sub-optimization episode |
| `-t`, `--test` / `--no-test`       | `bool`             | `True`                      | Whether to run in test mode                        |
| `-c`, `--compare` / `--no-compare` | `bool`             | `False`                     | Whether to compare against standalone optimizers   |
| `-e`, `--wandb_entity`             | `str` (Optional)   | `None`                      | Weights and Biases entity name                     |
| `-w`, `--wandb_project`            | `str` (Optional)   | `None`                      | Weights and Biases project name                    |

## Acknowledgment

Much of the implementation is adapted from:  
🔗 [Evolutionary-Intelligence/pypop](https://github.com/Evolutionary-Intelligence/pypop/tree/main)

---
## TO BE DONE

* Add more sub-optimizers (OPOA methods etc.)

* Enable larger portfolio (size 4 and 5).

* Enable additional parameters specifying agent's model architecture.