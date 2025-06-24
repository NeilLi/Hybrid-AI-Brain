import numpy as np
from src.core.match_score import match_score

def test_perfect_match():
    agent = np.array([1.0, 0.0, 0.0])
    task = np.array([1.0, 0.0, 0.0])
    score = match_score(agent, task)
    assert np.isclose(score, 1.0)

def test_orthogonal_vectors():
    agent = np.array([1.0, 0.0, 0.0])
    task = np.array([0.0, 1.0, 0.0])
    score = match_score(agent, task)
    assert np.isclose(score, 0.0)

def test_partial_overlap():
    agent = np.array([0.7, 0.7, 0.0])
    task = np.array([1.0, 0.0, 0.0])
    score = match_score(agent, task)
    assert score > 0 and score < 1

def test_zero_vectors():
    agent = np.zeros(3)
    task = np.zeros(3)
    score = match_score(agent, task)
    assert score == 0

def test_non_normalized_vectors():
    agent = np.array([3.0, 4.0, 0.0])
    task = np.array([0.0, 5.0, 12.0])
    score = match_score(agent, task)
    assert 0 <= score <= 1


