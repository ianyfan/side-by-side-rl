from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass

from gymnasium import Env
import torch

from common import Algorithm, MLPActorCritic


@dataclass
class Batch:
    observations: torch.Tensor
    actions: torch.Tensor
    old_log_probs: torch.Tensor
    advantages: torch.Tensor
    returns: torch.Tensor
    batch_size: int

    def __iter__(self) -> Iterator[Batch]:
        order = torch.randperm(len(self.observations))
        for start in range(0, len(order), self.batch_size):
            end = start + self.batch_size
            indices = order[start:end]

            yield Batch(
                observations=self.observations[indices],
                actions=self.actions[indices],
                old_log_probs=self.old_log_probs[indices],
                advantages=self.advantages[indices],
                returns=self.returns[indices],
                batch_size=self.batch_size
            )


@dataclass
class PPO(Algorithm):
    def initialize_networks(self) -> torch.nn.Module:
        return MLPActorCritic(self.env.observation_space,
                              self.env.action_space)

    # LINE 0
    def learn(
        self,
        learning_rate: float,
        horizon: int,  # T
        discount_rate: float,  # γ
        advantage_smoothing_factor: float,  # λ
        policy_iteration_steps: int,  # K
        batch_size: int,  # M
        clip_threshold: float,  # ε
        value_loss_importance: float,  # c₁
        entropy_loss_importance: float,  # c₂
    ) -> None:
        # LINE
        assert horizon % batch_size == 0

        self.policy = self.initialize_networks()

        policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            maximize=True
        )
        # LINE 1
        while True:
            # LINE 2
            # one actor
            # LINE 3
            rollout = self.generate_rollout(horizon)
            # LINE 4
            # Generalized advantage estimation
            with torch.no_grad():
                values = self.policy.value(
                    rollout.observations
                )
                next_values = self.policy.value(
                    rollout.next_observations
                )

            temporal_differences = (
                rollout.rewards
                + (
                    (1 - rollout.dones)
                    * discount_rate
                    * next_values
                )
                - values
            )

            advantages = torch.empty(horizon)
            last_advantage = 0
            for t in reversed(range(horizon)):
                if rollout.dones[t]:
                    last_advantage = 0
                last_advantage = advantages[t] = (
                    temporal_differences[t]
                    + discount_rate
                    * advantage_smoothing_factor
                    * last_advantage
                )
            # LINE 5
            # LINE
            returns = advantages + values

            with torch.no_grad():
                old_log_probs = self.policy.distribution(
                    rollout.observations
                ).log_prob(
                    rollout.actions
                )
            # LINE 6
            for _ in range(policy_iteration_steps):
                for batch in Batch(
                    observations=rollout.observations,
                    actions=rollout.actions,
                    old_log_probs=old_log_probs,
                    advantages=advantages,
                    returns=returns,
                    batch_size=batch_size
                ):
                    # LINE 7
                    log_probs = self.policy.distribution(
                        batch.observations
                    ).log_prob(
                        batch.actions
                    )
                    ratio = torch.exp(
                        log_probs - batch.old_log_probs
                    )
                    # LINE 8
                    policy_loss = torch.min(
                        ratio * batch.advantages,
                        torch.clamp(
                            ratio,
                            1 - clip_threshold,
                            1 + clip_threshold
                        )
                        * batch.advantages
                    ).mean()
                    # LINE 9
                    new_values = self.policy.value(
                        batch.observations
                    )
                    value_loss = torch.mean(
                        (new_values - batch.returns) ** 2
                    )
                    # LINE 10
                    entropy_loss = -log_probs.mean()
                    # LINE 11
                    loss = (
                        policy_loss
                        - value_loss_importance * value_loss
                        - entropy_loss_importance * entropy_loss
                    )
                    # LINE 12
                    policy_optimizer.zero_grad()
                    loss.backward()
                    policy_optimizer.step()
                    # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = PPO(env, callback=callback)
    model.learn(
        learning_rate=3e-4,
        horizon=2048,
        discount_rate=0.99,
        advantage_smoothing_factor=0.95,
        policy_iteration_steps=10,
        batch_size=64,
        clip_threshold=0.2,
        value_loss_importance=0.5,
        entropy_loss_importance=0.01
    )
