from experiments.scalability_study import calculate_processing_time
import numpy as np

def test_scalability_optimal_swarm_size():
    params = {
        "T_single": 10.0,
        "O_coord": 0.5,
        "c_comm": 0.1,
    }
    agent_counts = np.arange(1, 21)
    times = calculate_processing_time(agent_counts, **params, realistic=True)
    optimal_n = agent_counts[np.argmin(times)]
    assert optimal_n == 6  # Paper claims n=6 is optimal

