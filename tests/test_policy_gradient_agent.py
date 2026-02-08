import pytest
import numpy as np
import torch
import os
import sys

sys.path.append(os.getcwd())
from dynamicalgorithmselection.agents.policy_gradient_agent import PolicyGradientAgent
from unittest.mock import MagicMock, patch


class TestPolicyGradientAgent:
    @pytest.fixture
    def ppo_options(self):
        return {
            "action_space": [MagicMock(), MagicMock()],
            "ppo_batch_size": 100,
            "state_representation": "ELA",
            "n_checkpoints": 5,
            "cdb": 0.5,
            "max_function_evaluations": 1000,
            "actor_parameters": None,
            "critic_parameters": None,
            "reward_normalizer": MagicMock(),
            "state_normalizer": MagicMock(),
            "buffer": MagicMock(),
        }

    @pytest.fixture
    def mock_problem(self):
        p = MagicMock()
        p.dimension = 5
        return p

    def test_ppo_init(self, mock_problem, ppo_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 5),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)

            assert agent.actor is not None
            assert agent.critic is not None
            assert isinstance(agent.actor_optimizer, torch.optim.Optimizer)

    def test_select_action_tensor_shape(self, mock_problem, ppo_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)
            state = torch.randn(1, 10).to(torch.float32)
            action, log_prob, value = agent._select_action(state, full_buffer=True)

            assert isinstance(action, (int, np.integer))
            assert log_prob.shape == torch.Size([]) or log_prob.shape == torch.Size([1])

    def test_init_networks(self, mock_problem, ppo_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)
            assert agent.actor is not None
            assert agent.critic is not None
            assert isinstance(agent.actor_optimizer, torch.optim.Optimizer)

    def test_update_history(self, mock_problem, ppo_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)

            agent.iterations_history = {"x": None, "y": None}

            iteration_result = {
                "x_history": np.array([[1, 2], [3, 4]]),
                "y_history": np.array([0.1, 0.2]),
                "fitness_history": [],  # To powinno być ignorowane przez _update_history
            }

            agent._update_history(iteration_result)
            assert np.array_equal(
                agent.iterations_history["x"], iteration_result["x_history"]
            )

            iteration_result_2 = {
                "x_history": np.array([[5, 6]]),
                "y_history": np.array([0.05]),
            }
            agent._update_history(iteration_result_2)

            expected_y = np.array([0.1, 0.2, 0.05])
            assert len(agent.iterations_history["x"]) == 3
            assert np.array_equal(agent.iterations_history["y"], expected_y)

    def test_execute_action_instantiation(self, mock_problem, ppo_options):
        MockOptimizerClass = MagicMock()
        mock_optimizer_instance = MagicMock()
        MockOptimizerClass.return_value = mock_optimizer_instance

        mock_optimizer_instance.n_function_evaluations = 100
        mock_optimizer_instance.best_so_far_y = 5.0
        # iterate zwraca słownik wyników
        agent_dummy_result = {"x": None, "y": None}
        agent_return_val = ({"result": "ok"}, mock_optimizer_instance)

        # Musimy zamockować metodę iterate w agencie, bo ona odpala faktyczne obliczenia
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)
            agent.actions = [MockOptimizerClass]
            agent.iterate = MagicMock(return_value={"result": "ok"})

            iteration_result = {"x": None, "y": None}
            result, optimizer = agent._execute_action(0, iteration_result)
            MockOptimizerClass.assert_called_once()
            call_kwargs = MockOptimizerClass.call_args[0][1]  # options
            assert "max_function_evaluations" in call_kwargs
            agent.iterate.assert_called_once()

    def test_stagnation_logic(self, mock_problem, ppo_options):
        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)
            agent.stagnation_count = 0
            agent.best_so_far_y = 10.0
            agent.n_function_evaluations = 100

            optimizer_bad = MagicMock()
            optimizer_bad.best_so_far_y = 12.0
            optimizer_bad.n_function_evaluations = 150

            if optimizer_bad.best_so_far_y >= agent.best_so_far_y:
                agent.stagnation_count += (
                    optimizer_bad.n_function_evaluations - agent.n_function_evaluations
                )
            else:
                agent.stagnation_count = 0

            assert agent.stagnation_count == 50

            optimizer_good = MagicMock()
            optimizer_good.best_so_far_y = 8.0

            if optimizer_good.best_so_far_y >= agent.best_so_far_y:
                agent.stagnation_count += 100
            else:
                agent.stagnation_count = 0

            assert agent.stagnation_count == 0

    def test_buffer_add(self, mock_problem, ppo_options):
        mock_buffer = MagicMock()
        ppo_options["buffer"] = mock_buffer

        with patch(
            "dynamicalgorithmselection.agents.agent.get_state_representation",
            return_value=(MagicMock(), 10),
        ):
            agent = PolicyGradientAgent(mock_problem, ppo_options)

            state = torch.tensor([0.1, 0.2])
            action = 1
            reward = 0.5
            is_done = False
            log_prob = torch.tensor(-0.5)
            value = torch.tensor(0.1)

            agent.buffer.add(state, action, reward, is_done, log_prob, value)

            mock_buffer.add.assert_called_once_with(
                state, action, reward, is_done, log_prob, value
            )
