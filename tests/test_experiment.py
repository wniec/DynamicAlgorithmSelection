import unittest
from unittest.mock import MagicMock, patch, mock_open
from typing import Any, cast, Type  # Added imports

from dynamicalgorithmselection.experiments.experiment import (
    coco_bbob_experiment,
    run_comparison,
    dump_extreme_stats,
)
from dynamicalgorithmselection.optimizers.Optimizer import (
    Optimizer,
)  # Import the base class


class TestExperiment(unittest.TestCase):
    def setUp(self):
        self.optimizer_mock = MagicMock(
            spec=Type[Optimizer]
        )  # Use spec for better type safety
        self.optimizer_mock.__name__ = "MockOpt"
        self.options = {
            "name": "experiment_test",
            "max_function_evaluations": 100,
            "train_mode": False,
            "verbose": False,
        }
        self.eval_mult = 100

    @patch("dynamicalgorithmselection.experiments.experiment.run_cross_validation")
    def test_coco_bbob_experiment_dispatch_cv(self, mock_cv):
        coco_bbob_experiment(
            self.optimizer_mock, self.options, "test_exp", mode="CV_LOIO"
        )
        mock_cv.assert_called_once()
        self.assertTrue(mock_cv.call_args[1]["is_loio"])

    @patch("dynamicalgorithmselection.experiments.experiment._coco_bbob_test_all")
    def test_coco_bbob_experiment_dispatch_random(self, mock_test_all):
        coco_bbob_experiment(
            self.optimizer_mock, self.options, "test_exp", agent="random"
        )
        mock_test_all.assert_called_once()

    @patch("dynamicalgorithmselection.experiments.experiment.run_comparison")
    def test_coco_bbob_experiment_dispatch_baselines(self, mock_comparison):
        opts = self.options.copy()
        opts["baselines"] = True
        opts["optimizer_portfolio"] = [MagicMock()]
        coco_bbob_experiment(self.optimizer_mock, opts, "test_exp")
        mock_comparison.assert_called_once()

    @patch("dynamicalgorithmselection.experiments.experiment._coco_bbob_test")
    def test_coco_bbob_experiment_dispatch_test_only(self, mock_test):
        coco_bbob_experiment(self.optimizer_mock, self.options, "test_exp", train=False)
        mock_test.assert_called_once()

    @patch(
        "dynamicalgorithmselection.experiments.experiment._coco_bbob_neuroevolution_train"
    )
    def test_coco_bbob_experiment_dispatch_neuro(self, mock_neuro):
        coco_bbob_experiment(
            self.optimizer_mock, self.options, "test_exp", agent="neuroevolution"
        )
        mock_neuro.assert_called_once()

    @patch(
        "dynamicalgorithmselection.experiments.experiment._coco_bbob_policy_gradient_train"
    )
    def test_coco_bbob_experiment_dispatch_pg(self, mock_pg):
        coco_bbob_experiment(
            self.optimizer_mock, self.options, "test_exp", agent="policy-gradient"
        )
        mock_pg.assert_called_once()

    @patch("dynamicalgorithmselection.experiments.experiment.dump_extreme_stats")
    @patch("dynamicalgorithmselection.experiments.experiment.dump_stats")
    @patch("dynamicalgorithmselection.experiments.experiment.coco_bbob_single_function")
    @patch("dynamicalgorithmselection.experiments.experiment.cocoex")
    @patch("dynamicalgorithmselection.experiments.experiment.get_suite")
    @patch("dynamicalgorithmselection.experiments.experiment.os.makedirs")
    def test_run_comparison(
        self,
        mock_makedirs,
        mock_get_suite,
        mock_cocoex,
        mock_single_func,
        mock_dump_stats,
        mock_dump_extreme,
    ):
        # Setup mocks
        opt1 = MagicMock()
        opt1.__name__ = "Opt1"
        opt2 = MagicMock()
        opt2.__name__ = "Opt2"
        portfolio = cast(list[Type[Optimizer]], [opt1, opt2])

        # Mock Suite
        mock_suite_obj = MagicMock()
        mock_problem = MagicMock()
        mock_problem.dimension = 2
        mock_suite_obj.get_problem.return_value = mock_problem

        mock_get_suite.return_value = (mock_suite_obj, ["p1"])

        mock_single_func.return_value = {"fitness_history": [1, 2]}

        # Execute
        run_comparison(portfolio, self.options, 100)

        # Assertions
        self.assertEqual(mock_cocoex.Observer.call_count, 2)
        self.assertEqual(mock_single_func.call_count, 2)
        self.assertEqual(mock_dump_stats.call_count, 2)
        mock_dump_extreme.assert_called_once()

    @patch("dynamicalgorithmselection.experiments.experiment.get_checkpoints")
    @patch("dynamicalgorithmselection.experiments.experiment.get_extreme_stats")
    @patch("builtins.open", new_callable=mock_open)
    @patch("json.dump")
    def test_dump_extreme_stats(
        self, mock_json_dump, mock_file, mock_get_extreme, mock_get_checkpoints
    ):
        stats: dict[str, list[Any]] = {"Opt1": [], "Opt2": []}
        portfolio = cast(
            list[Type[Optimizer]],
            [MagicMock(__name__="Opt1"), MagicMock(__name__="Opt2")],
        )

        mock_get_extreme.return_value = ({"best": 1}, {"worst": 0})

        dump_extreme_stats(portfolio, stats, "p1", 100, 5, 10, 0.5)

        self.assertEqual(mock_file.call_count, 2)
        self.assertEqual(mock_json_dump.call_count, 2)

        args_list = mock_file.call_args_list
        self.assertIn("Opt1_Opt2_best", args_list[0][0][0])
        self.assertIn("Opt1_Opt2_worst", args_list[1][0][0])
