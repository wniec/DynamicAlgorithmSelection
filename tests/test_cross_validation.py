import unittest
from unittest.mock import MagicMock, patch
import os

import numpy as np

from dynamicalgorithmselection.experiments.utils import DIMENSIONS
from dynamicalgorithmselection.experiments.cross_validation import (
    run_cross_validation,
    _get_cv_folds,
)
from dynamicalgorithmselection.main import set_seed


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

    @patch(
        "dynamicalgorithmselection.experiments.cross_validation.cocoex.utilities.MiniPrint"
    )
    @patch("dynamicalgorithmselection.experiments.cross_validation.cocoex.Suite")
    def test_lodo_folds_only_contain_valid_dimensions(self, mock_suite, mock_miniprint):
        dims = DIMENSIONS  # [2, 3, 5, 10, 20, 40]
        n_folds = 3  # LODO uses 3 folds

        np.random.seed(42)
        suite, folds = _get_cv_folds(n_folds, leaving_mode="LODO", dim=dims)

        valid_dims_set = set(dims)

        for fold_idx, (train_set, test_set) in enumerate(folds):
            # Extract dimensions from problem IDs in the test set
            test_dims = set()
            for pid in test_set:
                # Format: bbob_f{fid}_i{iid}_d{dim}
                dim_str = pid.split("_d")[1]
                test_dims.add(int(dim_str))

            # Every dimension found in test set must be a valid BBOB dimension
            invalid_dims = test_dims - valid_dims_set
            self.assertEqual(
                invalid_dims,
                set(),
                f"Fold {fold_idx} test set contains invalid dimensions {invalid_dims}. "
                f"Valid dimensions are {valid_dims_set}. "
                f"This indicates remaining_dimensions was corrupted by function IDs.",
            )

            # Test set should not be empty
            self.assertGreater(
                len(test_set),
                0,
                f"Fold {fold_idx} has an empty test set.",
            )

        # All dimensions should be covered across all test folds
        all_test_dims = set()
        for _, test_set in folds:
            for pid in test_set:
                dim_str = pid.split("_d")[1]
                all_test_dims.add(int(dim_str))
        self.assertEqual(
            all_test_dims,
            valid_dims_set,
            f"Not all dimensions covered by LODO folds. "
            f"Covered: {all_test_dims}, Expected: {valid_dims_set}",
        )

    @patch(
        "dynamicalgorithmselection.experiments.cross_validation.cocoex.utilities.MiniPrint"
    )
    @patch("dynamicalgorithmselection.experiments.cross_validation.cocoex.Suite")
    def test_get_cv_folds_deterministic_after_set_seed(
        self, mock_suite, mock_miniprint
    ):
        for leaving_mode, dims in [
            ("LOIO", [10]),
            ("LOPO", [10]),
            ("LODO", DIMENSIONS),
        ]:
            n_folds = 3

            set_seed(42)
            _, folds_a = _get_cv_folds(n_folds, leaving_mode=leaving_mode, dim=dims)

            set_seed(42)
            _, folds_b = _get_cv_folds(n_folds, leaving_mode=leaving_mode, dim=dims)

            self.assertEqual(len(folds_a), len(folds_b))
            for i, ((train_a, test_a), (train_b, test_b)) in enumerate(
                zip(folds_a, folds_b)
            ):
                self.assertEqual(
                    sorted(train_a),
                    sorted(train_b),
                    f"mode={leaving_mode} fold {i}: train sets differ after identical seed",
                )
                self.assertEqual(
                    sorted(test_a),
                    sorted(test_b),
                    f"mode={leaving_mode} fold {i}: test sets differ after identical seed",
                )


if __name__ == "__main__":
    unittest.main()
