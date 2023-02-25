from collections.abc import Callable
from dataclasses import dataclass, field

import gymnasium
from gymnasium import Env
import torch

from common import Algorithm, MLPDeterministicActor, ReplayBuffer


@dataclass
class DQN(Algorithm):
    q_network: torch.nn.Module = field(init=False)

    def __post_init__(self) -> None:
        assert isinstance(self.env.action_space, gymnasium.spaces.Discrete)
        self.policy = self._policy

    def _policy(self, observation: torch.Tensor) -> torch.Tensor:
        return self.q_network(observation).argmax()

    def initialize_networks(self) -> torch.nn.Module:
        return MLPDeterministicActor(self.env.observation_space,
                                     self.env.action_space)

    # LINE 0
    def learn(
        self,
        replay_capacity: int,  # N
        learning_rate: float,
        exploration_rate: float,  # ε
        batch_size: int,
        discount_rate: float  # γ
    ) -> None:
        # LINE 1
        replay_memory = ReplayBuffer(replay_capacity)
        # LINE 2
        self.q_network = self.initialize_networks()
        # LINE
        q_network_optimizer = torch.optim.RMSprop(
            self.q_network.parameters(),
            lr=learning_rate
        )
        # LINE 3
        while True:
            # LINE 4
            observation = self.env_reset()
            # LINE 5
            done = False
            while not done:
                # LINE 6
                if torch.rand(1) < exploration_rate:
                    action = torch.as_tensor(
                        self.env.action_space.sample()
                    )
                # LINE 7
                else:
                    with torch.no_grad():
                        action = self.policy(observation)
                # LINE 8
                (
                    next_observation,
                    reward,
                    done
                ) = self.env_step(action)
                # LINE 9
                # use observations directly
                # instead of processing
                # LINE 10
                replay_memory.store(
                    (
                        observation,
                        action,
                        reward,
                        next_observation,
                        done
                    )
                )
                # LINE 11
                replay_data = replay_memory.sample(batch_size)
                # LINE 12
                target_q = (
                    replay_data.rewards
                    + (
                        (1 - replay_data.dones)
                        * discount_rate
                        * self.q_network(
                            replay_data.next_observations
                        ).max(dim=1).values
                    )
                )
                # LINE 13
                predicted_q = self.q_network(
                    replay_data.observations
                ).gather(
                    1, replay_data.actions.unsqueeze(1)
                ).squeeze()

                q_loss = torch.mean(
                    (target_q - predicted_q) ** 2
                )

                q_network_optimizer.zero_grad()
                q_loss.backward()
                q_network_optimizer.step()
                # LINE
                observation = next_observation
                # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = DQN(env, callback=callback)
    model.learn(
        replay_capacity=1000000,
        learning_rate=2.5e-4,
        exploration_rate=0.1,
        batch_size=32,
        discount_rate=1
    )
