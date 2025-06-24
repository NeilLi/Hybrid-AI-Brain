from experiments.memory_freshness import calculate_max_staleness

def test_memory_staleness_bound():
    params = {
        "lambda_d": 0.45,
        "W_max": 50.0,
        "lambda_t": 10.0,
        "c_bar": 0.8,
    }
    max_staleness = calculate_max_staleness(params)
    assert max_staleness < 3.0  # Must meet the <3s theoretical bound

