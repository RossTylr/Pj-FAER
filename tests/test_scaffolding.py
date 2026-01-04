"""Tests to verify project scaffolding is correct."""


def test_import_faer():
    """Verify faer package can be imported."""
    import faer

    assert faer.__version__ == "0.1.0"


def test_import_core_modules():
    """Verify core submodules can be imported."""
    from faer import core
    from faer.core import scenario
    from faer.core import arrivals
    from faer.core import distributions

    assert core is not None
    assert scenario is not None
    assert arrivals is not None
    assert distributions is not None


def test_import_model_modules():
    """Verify model submodules can be imported."""
    from faer import model
    from faer.model import patient
    from faer.model import resources
    from faer.model import processes

    assert model is not None
    assert patient is not None
    assert resources is not None
    assert processes is not None


def test_import_results_modules():
    """Verify results submodules can be imported."""
    from faer import results
    from faer.results import collector
    from faer.results import metrics

    assert results is not None
    assert collector is not None
    assert metrics is not None


def test_import_experiment_modules():
    """Verify experiment submodules can be imported."""
    from faer import experiment
    from faer.experiment import runner
    from faer.experiment import analysis

    assert experiment is not None
    assert runner is not None
    assert analysis is not None


def test_import_dependencies():
    """Verify key dependencies are installed."""
    import simpy
    import numpy as np
    import pandas as pd
    import scipy

    assert simpy is not None
    assert np is not None
    assert pd is not None
    assert scipy is not None


def test_import_sim_tools():
    """Verify sim-tools is installed and importable."""
    from sim_tools.distributions import Exponential

    assert Exponential is not None
