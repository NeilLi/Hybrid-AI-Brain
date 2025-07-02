def test_convergence_default():
    from benchmarks.convergence_validation import validate_probability
    res = validate_probability(n_trials=2000, spectral_norm=0.7, beta=1.0)
    assert res["prob_le_2"] >= 0.87
