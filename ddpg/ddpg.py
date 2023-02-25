from __future__ import annotations

from collections.abc import Callable, Mapping
import copy
from dataclasses import dataclass, field

from gymnasium import Env
import torch

from common import (Algorithm,
                    MLPActionCritic,
                    MLPDeterministicActor,
                    ReplayBuffer)


@dataclass
class OrnsteinUhlenbeckProcess:
    shape: tuple[int, ...]
    noise_scale: float  # θ
    normalization_strength: float  # σ
    steps: int = 100  # 1/dt

    value: torch.Tensor = field(init=False)

    def __post_init__(self) -> None:
        self.value = torch.zeros(self.shape)

    def __call__(self) -> torch.Tensor:
        timestep = 1 / self.steps
        for _ in range(self.steps):
            noise = torch.normal(0, 1, self.shape)
            self.value = (self.value
                          - self.normalization_strength * timestep * self.value
                          + self.noise_scale * (2 * timestep) ** 0.5 * noise)
        return self.value


@dataclass
class DDPG(Algorithm):
    def initialize_networks(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        state_space = self.env.observation_space
        action_space = self.env.action_space
        return (
            MLPDeterministicActor(state_space, action_space, tanh_output=True),
            MLPActionCritic(state_space, action_space)
        )

    def initialize_noise(self,
                         **noise_kwargs: float) -> Callable[[], torch.Tensor]:
        return OrnsteinUhlenbeckProcess(shape=self.env.action_space.shape,
                                        **noise_kwargs)

    # LINE 0
    def learn(
        self,
        learning_rate: float,
        critic_learning_rate: float,
        replay_size: int,
        noise_kwargs: Mapping[str, float],
        batch_size: int,
        discount_rate: float,  # γ
        target_network_update_rate: float  # τ
    ) -> None:
        # LINE 1
        (
            self.policy,
            self.critic
        ) = self.initialize_networks()
        # LINE
        policy_optimizer = torch.optim.Adam(
            self.policy.parameters(),
            lr=learning_rate,
            maximize=True
        )
        critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_learning_rate,
            weight_decay=0.01
        )
        # LINE 2
        target_policy = copy.deepcopy(self.policy)
        target_policy.requires_grad_(False)

        target_critic = copy.deepcopy(self.critic)
        target_critic.requires_grad_(False)
        # LINE 3
        replay_buffer = ReplayBuffer(replay_size)
        # LINE 4
        while True:
            # LINE 5
            noise = self.initialize_noise(**noise_kwargs)
            # LINE 6
            state = self.env_reset()
            # LINE 7
            done = False
            while not done:
                # LINE 8
                with torch.no_grad():
                    action = self.policy(state) + noise()
                # LINE 9
                next_state, reward, done = self.env_step(action)
                # LINE 10
                replay_buffer.store(
                    (state, action, reward, next_state, done)
                )
                # LINE 11
                batch = replay_buffer.sample(batch_size)
                # LINE 12
                target_value = (
                    batch.rewards
                    + discount_rate
                    * target_critic(
                        batch.next_states,
                        target_policy(batch.next_states)
                    )
                )
                # LINE 13
                predicted_value = self.critic(
                    batch.states,
                    batch.actions
                )
                critic_loss = torch.mean(
                    (target_value - predicted_value) ** 2
                )

                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()
                # LINE 14-15
                policy_loss = torch.mean(
                    self.critic(
                        batch.states,
                        self.policy(batch.states)
                    )
                )

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                # LINE 16-18
                for network, target_network in (
                    (self.policy, target_policy),
                    (self.critic, target_critic)
                ):
                    target_network.load_state_dict({
                        key: (
                            target_network_update_rate * val
                            + (1 - target_network_update_rate)
                            * target_network.state_dict()[key]
                        )
                        for key, val in network.state_dict().items()
                    })
                # LINE
                state = next_state
                # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = DDPG(env, callback=callback)
    model.learn(
        learning_rate=1e-4,
        critic_learning_rate=1e-3,
        replay_size=1000000,
        noise_kwargs={'noise_scale': 0.2, 'normalization_strength': 0.15},
        batch_size=64,
        discount_rate=0.99,
        target_network_update_rate=0.001
    )
