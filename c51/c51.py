from collections.abc import Callable
import copy
from dataclasses import dataclass, field

import gymnasium
from gymnasium import Env
import torch

from common import Algorithm, mlp, ReplayBuffer


class Q(torch.nn.Module):
    def __init__(self, input_size: int, output_size: int, atoms: int) -> None:
        super().__init__()
        self.q_net = mlp(input_size, output_size * atoms)
        self.output_size = output_size
        self.atoms = atoms

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = self.q_net(state)
        logits = x.reshape(*x.shape[:-1], self.output_size, self.atoms)
        return logits.softmax(-1)


@dataclass
class C51(Algorithm):
    q_net: torch.nn.Module = field(init=False)

    atoms: int = 51
    minimum_value: float = -100
    maximum_value: float = 100

    def __post_init__(self) -> None:
        assert isinstance(self.env.action_space, gymnasium.spaces.Discrete)
        self.policy = self._policy

        self.value_step = ((self.maximum_value - self.minimum_value)
                           / (self.atoms - 1))
        self.values = torch.arange(self.minimum_value,
                                   self.maximum_value + self.value_step,
                                   self.value_step)

    def _policy(self, observation: torch.Tensor) -> torch.Tensor:
        p = self.q_net(observation)
        q = (p * self.values).sum(-1)
        return q.argmax()

    def initialize_networks(self) -> torch.nn.Module:
        return Q(
            self.env.observation_space.shape[0],
            self.env.action_space.n,
            self.atoms
        )

    # LINE 0
    def learn(
        self,
        replay_capacity: int,
        learning_rate: float,
        epochs: int,
        epsilon: float,
        batch_size: int,
        discount_rate: float,
        target_network_update_frequency: int
    ) -> None:
        replay_memory = ReplayBuffer(replay_capacity)
        self.q_net = self.initialize_networks()
        q_net_optimizer = torch.optim.Adam(
            self.q_net.parameters(),
            lr=learning_rate
        )

        target_q_net = copy.deepcopy(self.q_net).requires_grad_(False)
        for _ in range(epochs):
            observation = torch.as_tensor(self.env.reset()[0])
            done = False
            while not done:
                if torch.rand(1) < epsilon:
                    action = torch.as_tensor(self.env.action_space.sample())
                else:
                    with torch.no_grad():
                        action = self.policy(observation)

                (
                    next_observation,
                    reward,
                    done
                ) = self.env_step(action)

                replay_memory.store(
                    (observation, action, reward, next_observation, done)
                )
                # LINE 1
                replay_data = replay_memory.sample(batch_size)
                # LINE 2
                next_q_probs = target_q_net(replay_data.next_observations)
                # LINE 3
                next_actions = (
                    self.values * next_q_probs
                ).sum(-1).argmax(-1)
                # LINE 4
                target_distribution = torch.zeros(self.atoms, batch_size)
                # LINE 5
                # batch
                # LINE 6
                # LINE 7
                target_values = torch.clamp(
                    replay_data.rewards
                    + (1 - replay_data.dones)
                    * discount_rate
                    * self.values.unsqueeze(1),
                    self.minimum_value,
                    self.maximum_value
                )
                # LINE 8
                target_values_positions = (
                    (target_values - self.minimum_value)
                    / self.value_step
                )
                # LINE 9
                target_indices_lower = target_values_positions.floor().long()
                target_indices_upper = target_values_positions.ceil().long()
                # LINE 10
                next_value_probs = next_q_probs.take_along_dim(
                    next_actions.reshape(-1, 1, 1),
                    1
                ).squeeze(1).T
                next_value_probs_lower = (
                    next_value_probs
                    # * (target_indices_upper - target_values_positions)
                    * (target_indices_lower + 1 - target_values_positions)
                )
                next_value_probs_upper = (
                    next_value_probs
                    * (target_values_positions - target_indices_lower)
                )
                for atom in range(self.atoms):
                    # LINE 11
                    target_distribution.scatter_add_(
                        0,
                        target_indices_lower[[atom]],
                        next_value_probs_lower[[atom]]
                    )
                    # LINE 12
                    target_distribution.scatter_add_(
                        0,
                        target_indices_upper[[atom]],
                        next_value_probs_upper[[atom]]
                    )
                # LINE
                target_distribution = target_distribution.T
                # LINE 13
                # LINE 14
                value = self.q_net(
                    replay_data.observations
                ).take_along_dim(
                    replay_data.actions.reshape(-1, 1, 1),
                    1,
                ).squeeze(1)
                q_loss = -(target_distribution * value.log()).sum()
                q_loss /= batch_size
                # LINE
                q_net_optimizer.zero_grad()
                q_loss.backward()
                q_net_optimizer.step()

                if self.timestep % target_network_update_frequency == 0:
                    target_q_net.load_state_dict(self.q_net.state_dict())

                observation = next_observation
                # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = C51(env, callback=callback)
    model.learn(
        epochs=10000,
        replay_capacity=1000000,
        learning_rate=2.5e-4,
        epsilon=0.05,
        batch_size=32,
        discount_rate=0.99,
        target_network_update_frequency=100
    )
