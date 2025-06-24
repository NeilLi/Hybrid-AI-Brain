from experiments.safety_bounds_test import calculate_hoeffding_bound

def test_hoeffding_safety_bound():
    n_samples = 59
    tau_safe = 0.7
    p_benign = 0.4
    epsilon = tau_safe - p_benign
    max_prob = calculate_hoeffding_bound(n_samples, epsilon)
    assert max_prob <= 1e-4  # Paper: Pr[false-block] <= 1e-4

