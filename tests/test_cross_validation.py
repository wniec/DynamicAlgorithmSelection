import unittest
from unittest.mock import MagicMock, patch
import os

from dynamicalgorithmselection.experiments.cross_validation import (
    run_cross_validation,
    _get_cv_folds,
)


class TestCrossValidation(unittest.TestCase):
    def setUp(self):
        self.optimizer_mock = MagicMock()
        self.options = {"name": "test_cv_opt"}
        self.eval_mult = 10

    @patch(
        "dynamicalgorithmselection.experiments.cross_validation.cocoex.utilities.MiniPrint"
    )
    @patch("dynamicalgorithmselection.experiments.cross_validation.cocoex.Suite")
    def test_get_cv_folds_structure(self, mock_suite, mock_miniprint):
        # Test internal logic of fold generation
        # We rely on the real imports for ALL_FUNCTIONS etc from utils,
        # so we check if it returns lists of correct length/structure.

        n_folds = 4
        suite, folds = _get_cv_folds(n_folds, leaving_mode="LOIO", dim=[10])

        self.assertIsInstance(suite, MagicMock)  # Should return the mocked suite
        self.assertEqual(len(folds), n_folds)

        for train, test in folds:
            self.assertIsInstance(train, list)
            self.assertIsInstance(test, list)
            # Intersection should be empty (no leakage)
            self.assertTrue(set(train).isdisjoint(set(test)))

    @patch("dynamicalgorithmselection.experiments.cross_validation.run_testing")
    @patch("dynamicalgorithmselection.experiments.cross_validation.run_training")
    @patch("dynamicalgorithmselection.experiments.cross_validation._get_cv_folds")
    @patch("dynamicalgorithmselection.experiments.cross_validation.cocoex.Observer")
    @patch("dynamicalgorithmselection.experiments.cross_validation.os.path.exists")
    @patch("dynamicalgorithmselection.experiments.cross_validation.os.mkdir")
    def test_run_cross_validation_flow(
        self,
        mock_mkdir,
        mock_exists,
        mock_observer,
        mock_get_folds,
        mock_run_training,
        mock_run_testing,
    ):
        # Setup mocks
        mock_exists.return_value = False  # Force directory creation
        mock_suite = MagicMock()

        # Create fake folds: 2 folds, simple IDs
        fake_folds = [
            (["p1", "p2"], ["p3"]),  # Fold 1: Train on p1,p2, Test on p3
            (["p3", "p4"], ["p1"]),  # Fold 2: Train on p3,p4, Test on p1
        ]
        mock_get_folds.return_value = (mock_suite, fake_folds)

        observer_instance = MagicMock()
        observer_instance.result_folder = "results/test_cv_opt"
        mock_observer.return_value = observer_instance

        # Execute
        res_folder = run_cross_validation(
            self.optimizer_mock, self.options, self.eval_mult, leaving_mode="LOIO"
        )

        # Assertions
        mock_mkdir.assert_called_with(os.path.join("results", "test_cv_opt"))

        # Check if training and testing were called for each fold
        self.assertEqual(mock_run_training.call_count, 2)
        self.assertEqual(mock_run_testing.call_count, 2)

        # Check clean up of options
        self.assertNotIn("buffer", self.options)
        self.assertNotIn("model_parameters", self.options)

        # Check return value
        self.assertEqual(res_folder, "results/test_cv_opt")


if __name__ == "__main__":
    unittest.main()
