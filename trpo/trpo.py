from collections.abc import Callable
from dataclasses import dataclass

from gymnasium import Env
import torch

from common import Algorithm, Rollout, MLPActorCriticPolicy


@dataclass
class TRPO(Algorithm):
    def initialize_policy(self) -> torch.nn.Module:
        return MLPActorCriticPolicy(self.env.observation_space.shape[0],
                                    self.env.action_space.n)

    def returns(self, episode: Rollout, discount_rate: float) -> torch.Tensor:
        returns = [0]
        for reward in reversed(episode.rewards):
            returns.append(reward + discount_rate * returns[-1])
        return torch.as_tensor(returns[:0:-1])

    # LINE 0
    def learn(
        self,
        learning_rate: float,
        epochs: int,
        discount_rate: float,
        maximum_policy_improvement_steps: int,
        kl_divergence_threshold: float
    ) -> None:
        # LINE 1
        self.policy = self.initialize_policy()
        # LINE
        policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            maximize=True
        )
        value_loss = torch.nn.MSELoss()
        # LINE 2
        for _ in range(epochs):
            # LINE
            with torch.no_grad():
                episode = self.generate_episode()
                old_log_probs = self.policy.distribution(
                    episode.observations
                ).log_prob(
                    episode.actions
                )
            # LINE 3
                returns = self.returns(episode, discount_rate)
                values = self.policy.value(episode.observations)
                advantages = returns - values
            # LINE 4-8
            # LINE 9
            # train the policy subject to the KL divergence constraint
            # by taking multiple steps and checking the condition at each step
            for _ in range(maximum_policy_improvement_steps):
                policy_optimizer.zero_grad()
                log_probs = self.policy.distribution(
                    episode.observations
                ).log_prob(
                    episode.actions
                )
                log_ratio = log_probs - old_log_probs
                ratio = torch.exp(log_ratio)
                loss = (ratio * advantages).mean()

                # extra term to train the critic
                loss -= value_loss(
                    self.policy.value(episode.observations).squeeze(),
                    returns
                )

                loss.backward()
                policy_optimizer.step()
                # LINE 10
                mean_kl_divergence_estimate = -log_ratio.mean()
                if mean_kl_divergence_estimate > kl_divergence_threshold:
                    break
                # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[TRPO], None] | None = None) -> None:
    model = TRPO(env, callback=callback)
    model.learn(
        learning_rate=1e-3,
        epochs=10000,
        discount_rate=1,
        maximum_policy_improvement_steps=100,
        kl_divergence_threshold=0.01
    )
