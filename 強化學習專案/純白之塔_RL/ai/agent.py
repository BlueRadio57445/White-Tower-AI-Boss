"""
PPO Agent with hybrid action space.
Uses squared probability distribution for discrete actions (hardware-friendly).
"""

from typing import Tuple, List, Optional, Dict, Any
import numpy as np

from core.math_utils import squared_prob, gaussian_log_prob


class HybridPPOAgent:
    """
    Proximal Policy Optimization agent with hybrid action space.

    Uses squared probability distribution for discrete actions to avoid
    dead neuron problems. Supports both discrete and continuous actions.

    Action Space:
        Discrete (7 actions): 0=FORWARD, 1=BACKWARD, 2=LEFT, 3=RIGHT, 4=OUTER_SLASH, 5=MISSILE, 6=HAMMER
        Continuous (2 actors): aim_missile (actor 0), aim_hammer (actor 1)

    Skill to Actor Mapping:
        4 (outer_slash): No aim required
        5 (missile): Uses aim_missile (actor 0)
        6 (hammer): Uses aim_hammer (actor 1)
    """

    # Skill action to aim actor mapping
    # -1 means no aim required, 0 = aim_missile, 1 = aim_hammer
    SKILL_TO_AIM_ACTOR = {
        4: -1,  # outer_slash - no aim
        5: 0,   # missile - aim_missile
        6: 1,   # hammer - aim_hammer
    }

    def __init__(
        self,
        n_features: int,
        n_discrete_actions: int = 7,
        n_aim_actors: int = 2,
        gamma: float = 0.99,
        lmbda: float = 0.95,
        epsilon: float = 0.2,
        sigma_init: float = 0.6,
        sigma_min: float = 0.15,
        sigma_decay: float = 0.9998
    ):
        """
        Initialize the PPO agent.

        Args:
            n_features: Number of input features
            n_discrete_actions: Number of discrete actions (default 7)
            n_aim_actors: Number of aim actors (default 2)
            gamma: Discount factor
            lmbda: GAE lambda parameter
            epsilon: PPO clipping parameter
            sigma_init: Initial exploration standard deviation
            sigma_min: Minimum exploration standard deviation
            sigma_decay: Exploration decay rate per update
        """
        self.n_features = n_features
        self.n_discrete_actions = n_discrete_actions
        self.n_aim_actors = n_aim_actors
        self.gamma = gamma
        self.lmbda = lmbda
        self.epsilon = epsilon

        # Exploration parameters
        self.sigma = sigma_init
        self.sigma_min = sigma_min
        self.sigma_decay = sigma_decay

        # Initialize weights with small random values
        self.w_actor_discrete = np.random.randn(n_discrete_actions, n_features) * 0.01
        self.w_critic = np.random.randn(n_features) * 0.01

        # Multiple aim actors (one for each skill that needs aiming)
        self.w_aim_actors = [
            np.random.randn(n_features) * 0.01
            for _ in range(n_aim_actors)
        ]

        # Legacy compatibility: keep w_actor_continuous_mu pointing to first aim actor
        self.w_actor_continuous_mu = self.w_aim_actors[0] if n_aim_actors > 0 else np.random.randn(n_features) * 0.01

        # Experience buffer
        self.buffer: List[tuple] = []

    def get_action(self, observation: np.ndarray) -> Tuple[int, List[float], float, List[float], float, np.ndarray]:
        """
        Select an action given the current observation.

        Args:
            observation: Feature vector

        Returns:
            Tuple of:
                - action_discrete: Selected discrete action index (0-6)
                - aim_values: List of aim values for each actor
                - prob_discrete: Probability of selected discrete action
                - mus: List of mu values for each aim actor
                - value: Estimated state value
                - logits: Raw discrete logits (for gradient computation)
        """
        # Discrete action selection using squared probability
        logits = np.dot(self.w_actor_discrete, observation)
        probs_discrete, scores, sum_scores, logits_saved = squared_prob(logits)
        action_discrete = np.random.choice(len(probs_discrete), p=probs_discrete)

        # Compute aim values for all actors
        mus = []
        aim_values = []
        for i in range(self.n_aim_actors):
            mu = np.dot(self.w_aim_actors[i], observation)
            mus.append(mu)
            aim_values.append(np.random.normal(mu, self.sigma))

        # Value estimation
        value = np.dot(self.w_critic, observation)

        return (
            action_discrete,
            aim_values,
            probs_discrete[action_discrete],
            mus,
            value,
            logits_saved
        )

    def get_aim_value_for_action(self, action_discrete: int, aim_values: List[float]) -> float:
        """
        Get the relevant aim value for a given discrete action.

        Args:
            action_discrete: The discrete action (0-6)
            aim_values: List of aim values from all actors

        Returns:
            The aim value to use (0.0 if action doesn't need aiming)
        """
        actor_idx = self.SKILL_TO_AIM_ACTOR.get(action_discrete, -1)
        if actor_idx >= 0 and actor_idx < len(aim_values):
            return aim_values[actor_idx]
        return 0.0

    def get_action_deterministic(self, observation: np.ndarray) -> Tuple[int, List[float]]:
        """
        Select action deterministically (for evaluation).

        Args:
            observation: Feature vector

        Returns:
            Tuple of (discrete_action, aim_values)
        """
        logits = np.dot(self.w_actor_discrete, observation)
        probs_discrete, _, _, _ = squared_prob(logits)
        action_discrete = np.argmax(probs_discrete)

        # Get mu for each aim actor
        aim_values = [np.dot(self.w_aim_actors[i], observation) for i in range(self.n_aim_actors)]
        return action_discrete, aim_values

    def store_transition(self, transition: tuple) -> None:
        """
        Store a transition in the experience buffer.

        Args:
            transition: Tuple of (obs, a_d, aim_values, prob_d, mus, value, logits, reward)
        """
        self.buffer.append(transition)

    def update(
        self,
        lr_actor_discrete: float = 0.003,
        lr_actor_continuous: float = 0.002,
        lr_critic: float = 0.007
    ) -> None:
        """
        Update agent weights using PPO with GAE.

        Args:
            lr_actor_discrete: Learning rate for discrete actor
            lr_actor_continuous: Learning rate for continuous actors
            lr_critic: Learning rate for critic
        """
        if not self.buffer:
            return

        # Unpack buffer
        (states, a_discretes, aim_values_list, old_probs_discrete,
         old_mus_list, values, old_logits_list, rewards) = zip(*self.buffer)

        # Compute GAE advantages
        advantages = self._compute_gae(rewards, values)
        returns = advantages + np.array(values)

        # Update weights for each transition
        for i in range(len(self.buffer)):
            state = states[i]
            a_d = a_discretes[i]
            aim_values = aim_values_list[i]
            target_v = returns[i]
            old_prob_d = old_probs_discrete[i]
            old_mus = old_mus_list[i]
            adv = advantages[i]

            # Update critic
            self._update_critic(state, target_v, lr_critic)

            # Update discrete actor
            self._update_actor_discrete(state, a_d, old_prob_d, adv, lr_actor_discrete)

            # Update all aim actors
            for actor_idx in range(self.n_aim_actors):
                if actor_idx < len(aim_values) and actor_idx < len(old_mus):
                    self._update_aim_actor(
                        state, aim_values[actor_idx], old_mus[actor_idx],
                        adv, lr_actor_continuous, actor_idx
                    )

        # Decay exploration
        self.sigma = max(self.sigma * self.sigma_decay, self.sigma_min)

        # Clear buffer
        self.buffer = []

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

    def _update_aim_actor(
        self, state: np.ndarray, action: float,
        old_mu: float, advantage: float, lr: float, actor_idx: int
    ) -> None:
        """Update aim actor weights using PPO clip."""
        if actor_idx >= self.n_aim_actors:
            return

        new_mu = np.dot(self.w_aim_actors[actor_idx], state)

        old_log_prob = gaussian_log_prob(action, old_mu, self.sigma)
        new_log_prob = gaussian_log_prob(action, new_mu, self.sigma)
        ratio = np.exp(new_log_prob - old_log_prob)

        clipped_ratio = np.clip(ratio, 1 - self.epsilon, 1 + self.epsilon)

        if (ratio * advantage <= clipped_ratio * advantage) or \
           (1 - self.epsilon < ratio < 1 + self.epsilon):
            grad = (action - new_mu) / (self.sigma ** 2) * state
            self.w_aim_actors[actor_idx] += lr * advantage * grad

    def _update_actor_continuous(
        self, state: np.ndarray, action: float,
        old_mu: float, advantage: float, lr: float
    ) -> None:
        """Legacy: Update first continuous actor weights using PPO clip."""
        self._update_aim_actor(state, action, old_mu, advantage, lr, 0)

    def get_weights(self) -> dict:
        """
        Get all agent weights.

        Returns:
            Dictionary of weight arrays
        """
        return {
            'w_actor_discrete': self.w_actor_discrete.copy(),
            'w_aim_actors': [w.copy() for w in self.w_aim_actors],
            'w_critic': self.w_critic.copy(),
            'sigma': self.sigma,
            'n_aim_actors': self.n_aim_actors
        }

    def set_weights(self, weights: dict) -> None:
        """
        Set agent weights from dictionary.

        Args:
            weights: Dictionary of weight arrays
        """
        if 'w_actor_discrete' in weights:
            self.w_actor_discrete = np.array(weights['w_actor_discrete'])
        if 'w_aim_actors' in weights:
            self.w_aim_actors = [np.array(w) for w in weights['w_aim_actors']]
            # Update legacy pointer
            if self.w_aim_actors:
                self.w_actor_continuous_mu = self.w_aim_actors[0]
        elif 'w_actor_continuous_mu' in weights:
            # Legacy support: convert old single actor to new format
            self.w_aim_actors = [np.array(weights['w_actor_continuous_mu'])]
            self.w_actor_continuous_mu = self.w_aim_actors[0]
        if 'w_critic' in weights:
            self.w_critic = np.array(weights['w_critic'])
        if 'sigma' in weights:
            self.sigma = weights['sigma']
