import unittest
from unittest.mock import MagicMock, patch
import copy

from dynamicalgorithmselection.experiments.core import run_testing, run_training


class TestCore(unittest.TestCase):
    def setUp(self):
        self.optimizer_mock = MagicMock()
        self.options = {
            "name": "test_experiment",
            "n_checkpoints": 5,
            "n_individuals": 10,
            "cdb": 0.5,
        }
        self.eval_multiplier = 10
        self.problem_ids = ["p1", "p2"]

        # Mock the Suite and Problem
        self.suite_mock = MagicMock()
        self.problem_mock = MagicMock()
        self.problem_mock.dimension = 2
        self.suite_mock.get_problem.return_value = self.problem_mock
        self.observer_mock = MagicMock()

    @patch(
        "dynamicalgorithmselection.experiments.core.tqdm",
        side_effect=lambda x, smoothing: x,
    )
    @patch("dynamicalgorithmselection.experiments.core.dump_stats")
    @patch("dynamicalgorithmselection.experiments.core.coco_bbob_single_function")
    def test_run_testing(self, mock_single_func, mock_dump_stats, mock_tqdm):
        # Setup return values
        mock_single_func.return_value = ({"fitness": 100}, {})

        run_testing(
            self.optimizer_mock,
            self.options,
            self.eval_multiplier,
            self.suite_mock,
            self.problem_ids,
            self.observer_mock,
        )

        # Assertions
        self.problem_mock.observe_with.assert_called_with(self.observer_mock)
        self.assertEqual(mock_single_func.call_count, 2)
        self.assertEqual(mock_dump_stats.call_count, 2)
        self.assertEqual(self.problem_mock.free.call_count, 2)

        expected_max_fe = self.eval_multiplier * self.problem_mock.dimension
        self.assertEqual(self.options["max_function_evaluations"], expected_max_fe)
        self.assertFalse(self.options["train_mode"])

    @patch(
        "dynamicalgorithmselection.experiments.core.tqdm",
        side_effect=lambda x, smoothing: x,
    )
    @patch("dynamicalgorithmselection.experiments.core.coco_bbob_single_function")
    def test_run_training(self, mock_single_func, mock_tqdm):
        fake_state_1 = {
            "state_normalizer": "norm1",
            "reward_normalizer": "rew1",
            "buffer": "buf1",
        }
        fake_state_2 = {
            "state_normalizer": "norm2",
            "reward_normalizer": "rew2",
            "buffer": "buf2",
        }

        # We use a list to store copies of options passed to the function
        captured_options = []

        def side_effect(optimizer, problem, options):
            # Capture a COPY of options at the moment of the call
            captured_options.append(copy.deepcopy(options))

            # Return different states based on how many times we've been called
            if len(captured_options) == 1:
                return ({"res": 1}, fake_state_1)
            return ({"res": 2}, fake_state_2)

        mock_single_func.side_effect = side_effect

        run_training(
            self.optimizer_mock,
            self.options,
            self.eval_multiplier,
            self.suite_mock,
            self.problem_ids,
        )

        self.assertEqual(mock_single_func.call_count, 2)

        # Verify the options passed to the SECOND call
        # The second call should have received 'norm1' because the first call returned fake_state_1
        second_call_options = captured_options[1]

        self.assertEqual(second_call_options["state_normalizer"], "norm1")
        self.assertEqual(second_call_options["buffer"], "buf1")
        self.assertTrue(second_call_options["train_mode"])
