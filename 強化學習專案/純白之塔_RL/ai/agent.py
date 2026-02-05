"""
PPO Agent with hybrid action space.
Uses squared probability distribution for discrete actions (hardware-friendly).
"""

from typing import Tuple, List, Optional
import numpy as np

from core.math_utils import squared_prob, gaussian_log_prob


class HybridPPOAgent:
    """
    Proximal Policy Optimization agent with hybrid action space.

    Uses squared probability distribution for discrete actions to avoid
    dead neuron problems. Supports both discrete and continuous actions.
    """

    def __init__(
        self,
        n_features: int,
        n_discrete_actions: int,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        epsilon: float = 0.2,
        sigma_init: float = 0.6,
        sigma_min: float = 0.1,
        sigma_max: float = 1.0,
        learnable_sigma: bool = True
    ):
        """
        Initialize the PPO agent.

        Args:
            n_features: Number of input features
            n_discrete_actions: Number of discrete actions
            gamma: Discount factor
            lmbda: GAE lambda parameter
            epsilon: PPO clipping parameter
            sigma_init: Initial exploration standard deviation
            sigma_min: Minimum sigma (for stability)
            sigma_max: Maximum sigma (for stability)
            learnable_sigma: If True, sigma is learned via policy gradient
        """
        self.n_features = n_features
        self.n_discrete_actions = n_discrete_actions
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon

        # Sigma parameters
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.learnable_sigma = learnable_sigma

        # Initialize weights with small random values
        self.w_actor_discrete = np.random.randn(n_discrete_actions, n_features) * 0.01
        self.w_actor_continuous_mu = np.random.randn(n_features) * 0.01
        self.w_critic = np.random.randn(n_features) * 0.01

        # Experience buffer
        self.buffer: List[tuple] = []

    def get_action(self, observation: np.ndarray) -> Tuple[int, float, float, float, float, np.ndarray]:
        """
        Select an action given the current observation.

        Args:
            observation: Feature vector

        Returns:
            Tuple of:
                - action_discrete: Selected discrete action index
                - action_continuous: Continuous action value
                - prob_discrete: Probability of selected discrete action
                - mu: Mean of continuous action distribution
                - value: Estimated state value
                - logits: Raw discrete logits (for gradient computation)
        """
        # Discrete action selection using squared probability
        logits = np.dot(self.w_actor_discrete, observation)
        probs_discrete, scores, sum_scores, logits_saved = squared_prob(logits)
        action_discrete = np.random.choice(len(probs_discrete), p=probs_discrete)

        # Continuous action selection
        mu = np.dot(self.w_actor_continuous_mu, observation)
        action_continuous = np.random.normal(mu, self.sigma)

        # Value estimation
        value = np.dot(self.w_critic, observation)

        return (
            action_discrete,
            action_continuous,
            probs_discrete[action_discrete],
            mu,
            value,
            logits_saved
        )

    def get_action_deterministic(self, observation: np.ndarray) -> Tuple[int, float]:
        """
        Select action deterministically (for evaluation).

        Args:
            observation: Feature vector

        Returns:
            Tuple of (discrete_action, continuous_action)
        """
        logits = np.dot(self.w_actor_discrete, observation)
        probs_discrete, _, _, _ = squared_prob(logits)
        action_discrete = np.argmax(probs_discrete)

        mu = np.dot(self.w_actor_continuous_mu, observation)
        return action_discrete, mu

    def store_transition(self, transition: tuple) -> None:
        """
        Store a transition in the experience buffer.

        Args:
            transition: Tuple of (obs, a_d, a_c, prob_d, mu, value, logits, reward)
        """
        self.buffer.append(transition)

    def update(
        self,
        lr_actor_discrete: float = 0.003,
        lr_actor_continuous: float = 0.002,
        lr_critic: float = 0.007,
        lr_sigma: float = 0.001
    ) -> None:
        """
        Update agent weights using PPO with GAE.

        Args:
            lr_actor_discrete: Learning rate for discrete actor
            lr_actor_continuous: Learning rate for continuous actor
            lr_critic: Learning rate for critic
            lr_sigma: Learning rate for sigma (if learnable)
        """
        if not self.buffer:
            return

        # Unpack buffer
        (states, a_discretes, a_continuouses, old_probs_discrete,
         old_mus, values, old_logits_list, rewards) = zip(*self.buffer)

        # Compute GAE advantages
        advantages = self._compute_gae(rewards, values)
        returns = advantages + np.array(values)

        # Accumulate sigma gradient over batch
        sigma_grad_accum = 0.0

        # Update weights for each transition
        for i in range(len(self.buffer)):
            state = states[i]
            a_d = a_discretes[i]
            a_c = a_continuouses[i]
            target_v = returns[i]
            old_prob_d = old_probs_discrete[i]
            old_mu = old_mus[i]
            adv = advantages[i]

            # Update critic
            self._update_critic(state, target_v, lr_critic)

            # Update discrete actor
            self._update_actor_discrete(state, a_d, old_prob_d, adv, lr_actor_discrete)

            # Update continuous actor (mu)
            self._update_actor_continuous(state, a_c, old_mu, adv, lr_actor_continuous)

            # Accumulate sigma gradient
            if self.learnable_sigma:
                new_mu = np.dot(self.w_actor_continuous_mu, state)
                sigma_grad_accum += self._compute_sigma_gradient(a_c, new_mu, adv)

        # Update sigma using accumulated gradient
        if self.learnable_sigma:
            avg_sigma_grad = sigma_grad_accum / len(self.buffer)
            avg_sigma_grad = np.clip(avg_sigma_grad, -1.0, 1.0)  # Clip for stability
            self.sigma += lr_sigma * avg_sigma_grad
            self.sigma = np.clip(self.sigma, self.sigma_min, self.sigma_max)

        # Clear buffer
        self.buffer = []

    def _compute_sigma_gradient(self, action: float, mu: float, advantage: float) -> float:
        """
        Compute policy gradient for sigma.

        Based on: d(log π)/d(σ) = ((a - μ)² - σ²) / σ³

        Args:
            action: The continuous action taken
            mu: Mean of the distribution
            advantage: The advantage estimate

        Returns:
            Gradient contribution for sigma
        """
        diff_sq = (action - mu) ** 2
        sigma_sq = self.sigma ** 2
        sigma_cubed = self.sigma ** 3

        # Gradient: advantage * d(log π)/d(σ)
        grad = advantage * (diff_sq - sigma_sq) / (sigma_cubed + 1e-8)
        return grad

    def _compute_gae(self, rewards: tuple, values: tuple) -> np.ndarray:
        """Compute Generalized Advantage Estimation."""
        advantages = []
        gae = 0
        next_v = 0

        for r, v in zip(reversed(rewards), reversed(values)):
            delta = r + self.gamma * next_v - v
            gae = delta + self.gamma * self.lmbda * gae
            advantages.insert(0, gae)
            next_v = v

        return np.array(advantages)

    def _update_critic(self, state: np.ndarray, target_v: float, lr: float) -> None:
        """Update critic weights using MSE loss."""
        current_v = np.dot(self.w_critic, state)
        grad = (target_v - current_v) * state
        grad = np.clip(grad, -5.0, 5.0)
        self.w_critic += lr * grad

    def _update_actor_discrete(
        self, state: np.ndarray, action: int,
        old_prob: float, advantage: float, lr: float
    ) -> None:
        """Update discrete actor weights using squared probability gradient."""
        logits_new = np.dot(self.w_actor_discrete, state)
        probs_new, scores_new, sum_new, _ = squared_prob(logits_new)

        ratio = probs_new[action] / (old_prob + 1e-8)
        clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

        # Only update if not clipped or within bounds
        if (ratio * advantage <= clipped_ratio * advantage) or \
           (1 - self.epsilon < ratio < 1 + self.epsilon):

            inv_sum = 1.0 / (sum_new + 1e-8)
            inv_prob = np.clip(1.0 / (probs_new[action] + 1e-8), 0, 50)

            # Gradient for selected action
            grad_selected = advantage * (2 * logits_new[action] * inv_sum) * (inv_prob - 1) * state
            self.w_actor_discrete[action] += lr * grad_selected

            # Gradient for other actions
            for j in range(self.n_discrete_actions):
                if j != action:
                    grad_other = -advantage * (2 * logits_new[j] * inv_sum) * state
                    self.w_actor_discrete[j] += lr * grad_other

    def _update_actor_continuous(
        self, state: np.ndarray, action: float,
        old_mu: float, advantage: float, lr: float
    ) -> None:
        """Update continuous actor weights using PPO clip."""
        new_mu = np.dot(self.w_actor_continuous_mu, state)

        old_log_prob = gaussian_log_prob(action, old_mu, self.sigma)
        new_log_prob = gaussian_log_prob(action, new_mu, self.sigma)
        ratio = np.exp(new_log_prob - old_log_prob)

        clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

        if (ratio * advantage <= clipped_ratio * advantage) or \
           (1 - self.epsilon < ratio < 1 + self.epsilon):
            grad = (action - new_mu) / (self.sigma ** 2) * state
            self.w_actor_continuous_mu += lr * advantage * grad

    def get_weights(self) -> dict:
        """
        Get all agent weights.

        Returns:
            Dictionary of weight arrays
        """
        return {
            'w_actor_discrete': self.w_actor_discrete.copy(),
            'w_actor_continuous_mu': self.w_actor_continuous_mu.copy(),
            'w_critic': self.w_critic.copy(),
            'sigma': self.sigma
        }

    def set_weights(self, weights: dict) -> None:
        """
        Set agent weights from dictionary.

        Args:
            weights: Dictionary of weight arrays
        """
        if 'w_actor_discrete' in weights:
            self.w_actor_discrete = np.array(weights['w_actor_discrete'])
        if 'w_actor_continuous_mu' in weights:
            self.w_actor_continuous_mu = np.array(weights['w_actor_continuous_mu'])
        if 'w_critic' in weights:
            self.w_critic = np.array(weights['w_critic'])
        if 'sigma' in weights:
            self.sigma = weights['sigma']
