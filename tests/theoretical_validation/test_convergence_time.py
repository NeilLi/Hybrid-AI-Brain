from experiments.convergence_analysis import calculate_expected_convergence_time
import numpy as np

def test_convergence_time_matches_theory():
    # Example probabilities from Section 8.1
    hop_probabilities = [0.89, 0.79, 0.92]
    expected_joint_prob = np.prod(hop_probabilities)
    expected_time = 1 / expected_joint_prob

    # Function should match the above calculation
    result = calculate_expected_convergence_time(hop_probabilities)
    assert abs(result - expected_time) < 1e-6
    assert result < 2.0  # Paper claims E[tau] <= 2 steps for this scenario

