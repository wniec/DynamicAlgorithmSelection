import numpy as np
import pytest

from dynamicalgorithmselection.optimizers.ES.CMAES import CMAES


@pytest.fixture
def problem():
    ndim = 5
    return {
        "fitness_function": lambda x: np.sum(x**2),
        "ndim_problem": ndim,
        "upper_boundary": np.ones(ndim) * 5,
        "lower_boundary": np.ones(ndim) * -5,
    }


@pytest.fixture
def options():
    return {
        "max_function_evaluations": 5000,
        "n_individuals": 10,
        "seed_rng": 42,
        "verbose": 0,
    }


@pytest.fixture
def cmaes(problem, options):
    return CMAES(problem, options)


class TestCMAESSetData:
    def test_none_input_resets_state(self, cmaes):
        cmaes.set_data(x=None, y=None)
        assert cmaes.start_conditions == {"x": None, "y": None, "mean": None}

    def test_preserves_mean_when_provided(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])
        mean = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

        cmaes.set_data(x=x, y=y, mean=mean)
        np.testing.assert_array_equal(cmaes.start_conditions["mean"], mean)

    def test_estimates_mean_when_not_provided(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])

        cmaes.set_data(x=x, y=y, mean=None)

        indices = np.argsort(y)[: cmaes.n_individuals]
        expected_mean = x[indices].mean(axis=0)
        np.testing.assert_array_almost_equal(
            cmaes.start_conditions["mean"], expected_mean
        )

    def test_preserves_sigma_when_provided(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])

        cmaes.set_data(x=x, y=y, sigma=0.42)
        assert cmaes.start_conditions["sigma"] == 0.42

    def test_estimates_sigma_when_not_provided(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])

        cmaes.set_data(x=x, y=y)
        assert cmaes.start_conditions["sigma"] > 0

    def test_preserves_cma_state(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])
        p_c = np.ones(5) * 0.1
        p_s = np.ones(5) * 0.2
        cm = np.eye(5) * 1.5
        e_ve = np.eye(5)
        e_va = np.ones(5) * 0.9

        cmaes.set_data(x=x, y=y, p_c=p_c, p_s=p_s, cm=cm, e_ve=e_ve, e_va=e_va)

        np.testing.assert_array_equal(cmaes.start_conditions["p_c"], p_c)
        np.testing.assert_array_equal(cmaes.start_conditions["p_s"], p_s)
        np.testing.assert_array_equal(cmaes.start_conditions["cm"], cm)
        np.testing.assert_array_equal(cmaes.start_conditions["e_ve"], e_ve)
        np.testing.assert_array_equal(cmaes.start_conditions["e_va"], e_va)

    def test_selects_best_individuals(self, cmaes):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])

        cmaes.set_data(x=x, y=y)

        indices = np.argsort(y)[: cmaes.n_individuals]
        np.testing.assert_array_equal(cmaes.start_conditions["x"], x[indices])
        np.testing.assert_array_equal(cmaes.start_conditions["y"], y[indices])

    def test_d_sliced_by_indices(self, cmaes):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])
        d = rng.standard_normal((20, 5))

        cmaes.set_data(x=x, y=y, d=d)

        indices = np.argsort(y)[: cmaes.n_individuals]
        np.testing.assert_array_equal(cmaes.start_conditions["d"], d[indices])

    def test_best_x_y_from_kwargs(self, cmaes):
        x = np.random.default_rng(0).standard_normal((20, 5))
        y = np.array([np.sum(xi**2) for xi in x])
        best_x = np.zeros(5)

        cmaes.set_data(x=x, y=y, best_x=best_x, best_y=0.5)

        np.testing.assert_array_equal(cmaes.best_so_far_x, best_x)
        assert cmaes.best_so_far_y == 0.5

    def test_list_y_coerced_to_ndarray(self, cmaes):
        rng = np.random.default_rng(0)
        x = rng.standard_normal((20, 5))
        y_list = [float(np.sum(xi**2)) for xi in x]

        cmaes.set_data(x=x, y=y_list)

        assert isinstance(cmaes.start_conditions["y"], np.ndarray)
        assert cmaes.start_conditions["x"].shape[0] == cmaes.n_individuals

    def test_roundtrip_preserves_state(self, cmaes, problem):
        """Run CMAES, extract state via get_data, feed back via set_data,
        and verify the internal state is preserved."""
        cmaes.target_FE = 50
        results = cmaes.optimize(fitness_function=problem["fitness_function"])

        data = cmaes.get_data()
        assert "sigma" in data, "get_data() should include sigma"

        cmaes2 = CMAES(problem, cmaes.options | {"seed_rng": 99})
        cmaes2.set_data(
            x=data.get("x"),
            y=data.get("y"),
            mean=data.get("mean"),
            p_c=data.get("p_c"),
            p_s=data.get("p_s"),
            cm=data.get("cm"),
            e_ve=data.get("e_ve"),
            e_va=data.get("e_va"),
            d=data.get("d"),
            sigma=data.get("sigma"),
            best_x=results.get("best_x"),
            best_y=results.get("best_y"),
        )

        sc = cmaes2.start_conditions
        np.testing.assert_array_equal(sc["mean"], data["mean"])
        assert sc["sigma"] == data["sigma"]
        np.testing.assert_array_equal(sc["p_c"], data["p_c"])
        np.testing.assert_array_equal(sc["p_s"], data["p_s"])
        np.testing.assert_array_equal(sc["cm"], data["cm"])
        np.testing.assert_array_equal(sc["e_ve"], data["e_ve"])
        np.testing.assert_array_equal(sc["e_va"], data["e_va"])
