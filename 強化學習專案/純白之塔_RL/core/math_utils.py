"""
Mathematical utilities for the reinforcement learning agent.
Hardware-friendly implementations using only basic arithmetic operations.
"""

import numpy as np


def squared_prob(logits: np.ndarray) -> tuple:
    """
    Hardware-friendly probability distribution using squared values.
    Avoids dead neuron problems by ensuring all values contribute.

    Formula:
        s_i = logit_i^2 + epsilon
        P_i = s_i / sum(s_k)

    Args:
        logits: Raw logit values from the network

    Returns:
        tuple: (probabilities, scores, total_score, original_logits)
            - probabilities: Normalized probability distribution
            - scores: Squared scores before normalization
            - total_score: Sum of all scores
            - logits: Original logits for gradient computation
    """
    scores = logits ** 2 + 1e-5  # Square + epsilon to avoid zeros
    total_score = np.sum(scores)
    probs = scores / total_score
    return probs, scores, total_score, logits


def gaussian_log_prob(x: float, mu: float, sigma: float) -> float:
    """
    Compute the log probability density of a Gaussian distribution.

    Args:
        x: The observed value
        mu: Mean of the distribution
        sigma: Standard deviation of the distribution

    Returns:
        float: Log probability density at x
    """
    var = sigma ** 2
    log_prob = -((x - mu) ** 2) / (2 * var) - np.log(sigma * np.sqrt(2 * np.pi))
    return log_prob
