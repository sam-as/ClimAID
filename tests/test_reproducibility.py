"""Models built by ClimAID run single-threaded, so results do not depend on the number of cores."""
import pytest

from climaid.model_registry import MODEL_REGISTRY, single_threaded


@pytest.mark.parametrize("name", ["rf", "extra_trees", "xgb", "lgbm"])
def test_threaded_models_are_forced_to_one_thread(name):
    if name not in MODEL_REGISTRY:
        pytest.skip(f"{name} not installed")
    assert single_threaded(MODEL_REGISTRY[name], {"n_jobs": -1})["n_jobs"] == 1


def test_catboost_uses_thread_count():
    if "catboost" not in MODEL_REGISTRY:
        pytest.skip("catboost not installed")
    params = single_threaded(MODEL_REGISTRY["catboost"], {"n_jobs": -1})
    assert params == {"thread_count": 1}


def test_models_without_threads_are_left_alone_and_input_not_mutated():
    params = {"alpha": 1.0}
    assert single_threaded(MODEL_REGISTRY["ridge"], params) == {"alpha": 1.0}
    rf = {"n_jobs": -1}
    single_threaded(MODEL_REGISTRY["rf"], rf)
    assert rf == {"n_jobs": -1}


def test_v2_estimators_are_single_threaded():
    from climaid.forecasting_v2.tuning import build_estimator
    est = build_estimator("random_forest")
    est = getattr(est, "steps", [[None, est]])[-1][1]
    assert est.get_params()["n_jobs"] == 1
