from tools.parameter_optimizer import ParameterOptimizer

def test_optimize_memory_decay_rate():
    params = {
        "W_max": 50.0,
        "lambda_t": 10.0,
        "c_bar": 0.8,
    }
    optimizer = ParameterOptimizer()
    target_t_f = 2.97
    lambda_d = optimizer.optimize_memory_decay_rate(target_t_f, params)
    # Verify that plugging lambda_d into the staleness equation gives t_f ~= 2.97
    import numpy as np
    t_f = (1 / lambda_d) * np.log(1 + (params["W_max"] * lambda_d) / (params["lambda_t"] * params["c_bar"]))
    assert abs(t_f - target_t_f) < 0.05

