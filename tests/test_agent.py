import pytest
import numpy as np
import os
import sys

sys.path.append(os.getcwd())

from dynamicalgorithmselection.agents.agent import Agent
from unittest.mock import MagicMock, patch


class TestAgent:
    @pytest.fixture
    def mock_problem(self):
        problem = MagicMock()
        problem.dimension = 10
        return problem

    @pytest.fixture
    def basic_options(self, mock_problem):
        return {
            "action_space": [MagicMock(__name__="Opt1"), MagicMock(__name__="Opt2")],
            "force_restarts": False,
            "name": "TestAgent",
            "cdb": 0.5,
            "n_checkpoints": 5,
            "max_function_evaluations": 1000,
            "n_individuals": 50,
            "state_representation": "ELA",
            "reward_normalizer": MagicMock(),
            "state_normalizer": MagicMock(),
        }

    def test_agent_initialization(self, mock_problem, basic_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation"
        ) as mock_get_sr:
            mock_sr_func = MagicMock(return_value=np.zeros(5))
            mock_get_sr.return_value = (mock_sr_func, 5)

            agent = Agent(mock_problem, basic_options)

            assert agent.name == "TestAgent"
            assert len(agent.actions) == 2
            assert agent.n_checkpoints == 5

    def test_get_reward_logic(self, mock_problem, basic_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 5),
        ):
            agent = Agent(mock_problem, basic_options)
            agent.initial_value_range = (10.0, 20.0)

            reward_good = agent.get_reward(new_best_y=15.0, old_best_y=20.0)
            # 5.0 / 10.0 = 0.5 -> log(0.5)
            assert np.isclose(reward_good, np.log(0.5 + 1e-5))

            reward_bad = agent.get_reward(new_best_y=25.0, old_best_y=20.0)
            assert np.isclose(reward_bad, np.log(0.0 + 1e-5))
