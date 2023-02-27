from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import ClassVar, Generic, NewType, SupportsFloat, TypeVar

import torch

StateT = TypeVar('StateT')
ActionT = TypeVar('ActionT')
RewardT = NewType('RewardT', SupportsFloat)


@dataclass
class EnvOutput(Generic[StateT]):
    state: StateT
    reward: RewardT = float('nan')
    done: bool = False


class Env(Generic[StateT, ActionT], ABC):
    state_type: ClassVar[type[StateT]]
    action_type: ClassVar[type[ActionT]]

    @abstractmethod
    def reset(self) -> EnvOutput[StateT]:
        """Reset environment."""

    @abstractmethod
    def step(self, action: ActionT) -> EnvOutput[StateT]:
        """Step enviroment."""


class CorridorState(Enum):
    GOAL = 0
    SQUARE = 1

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.value)])

    @classmethod
    @property
    def size(cls) -> int:
        return 1


class CorridorStep(Enum):
    LEFT = 0
    RIGHT = 1

    def as_tensor(self) -> torch.Tensor:
        return torch.tensor([float(self.value)])


@dataclass
class Corridor(Env[CorridorState, CorridorStep]):
    # 0 = start, 1 = switched, 2 = square, 3 = goal
    position: int = field(init=False)

    state_type: ClassVar[type[CorridorState]] = CorridorState
    action_type: ClassVar[type[CorridorStep]] = CorridorStep

    def reset(self) -> EnvOutput[CorridorState]:
        self.position = 0
        return EnvOutput(CorridorState.SQUARE)

    def step(self, action: CorridorStep) -> EnvOutput[CorridorState]:
        direction = -1 if action == CorridorStep.LEFT else 1

        if self.position == 0 and direction == -1:
            direction = 0
        elif self.position == 1:
            direction *= -1

        self.position += direction
        if self.position == 3:
            return EnvOutput(CorridorState.GOAL, -1, True)
        else:
            return EnvOutput(CorridorState.SQUARE, -1, False)


@dataclass
class Episode:
    states: list[torch.Tensor] = field(default_factory=list)
    actions: list[torch.Tensor] = field(default_factory=list)
    rewards: list[RewardT] = field(default_factory=lambda: [float('nan')])

    def __len__(self) -> int:
        return len(self.states)

    def add(self, state: StateT, action: ActionT, reward: RewardT) -> None:
        self.states.append(state.as_tensor())
        self.actions.append(action.as_tensor())
        self.rewards.append(reward)


class Linear(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.parameter = torch.nn.Parameter(torch.zeros(output_dim, input_dim))

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.parameter @ state_tensor


class LinearPolicy(torch.nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.linear = Linear(input_dim, output_dim)

    def distribution(
        self,
        state_tensor: torch.Tensor
    ) -> torch.distributions.Categorical:
        action_logits = self.linear(state_tensor)
        action_probs = torch.nn.functional.softmax(action_logits, dim=0)
        return torch.distributions.Categorical(action_probs)

    def forward(self, state_tensor: torch.Tensor) -> torch.Tensor:
        return self.distribution(state_tensor).sample()


@dataclass
class REINFORCE:
    env: Env
    discount_rate: float

    callback: Callable[['REINFORCE'], None] | None = None

    policy: torch.nn.Module = field(init=False)
    baseline: torch.nn.Module = field(init=False)

    def initialize_networks(self) -> tuple[torch.nn.Module, torch.nn.Module]:
        input_size = self.env.state_type.size
        return (
            LinearPolicy(input_size, len(self.env.action_type)),
            Linear(self.env.state_type.size, 1)
        )

    def generate_episode(self) -> Episode:
        episode = Episode()

        env_output = self.env.reset()
        while not env_output.done:
            state = env_output.state

            action_tensor = self.policy(state.as_tensor())
            action = self.env.action_type(action_tensor.item())

            env_output = self.env.step(action)
            episode.add(state, action, env_output.reward)

        return episode

    # LINE 0
    def learn(
        # LINE 1-2
        self,
        # LINE 3
        step_size: float,
        baseline_step_size: float,
        # LINE
    ) -> None:
        # LINE 4
        self.policy, self.baseline = self.initialize_networks()
        # LINE
        policy_optimizer = torch.optim.SGD(
            self.policy.parameters(),
            lr=step_size,
            maximize=True
        )
        value_optimizer = torch.optim.SGD(
            self.baseline.parameters(),
            lr=baseline_step_size
        )
        # LINE 5
        while True:
            # LINE 6
            episode = self.generate_episode()
            T = len(episode)
            # LINE 7
            for t in range(T):
                # LINE 8
                episode_return: float = sum(
                    self.discount_rate**(k-t-1) * episode.rewards[k]
                    for k in range(t + 1, T + 1)
                )
                # LINE 9
                advantage = (
                    episode_return
                    - self.baseline(episode.states[t])
                )
                # LINE 10
                value_loss = advantage ** 2

                value_optimizer.zero_grad()
                value_loss.backward()
                value_optimizer.step()
                # LINE 11
                policy_loss = (
                    self.discount_rate ** t
                    * advantage.detach()
                    * self.policy.distribution(
                        episode.states[t]
                    ).log_prob(
                        episode.actions[t]
                    )
                )

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                # END

            if self.callback is not None:
                self.callback(self)


if __name__ == '__main__':
    runs = 1
    epochs = 200
    performance = []

    def test_model(model) -> None:
        global epochs

        with torch.no_grad():
            p = model.policy.distribution(torch.tensor([1.])).probs[1]
            performance.append(-2 * (2 - p) / (p * (1 - p)))

        epochs -= 1
        if not epochs:
            import matplotlib.pyplot as plt
            plt.plot(performance)
            plt.show()

    env = Corridor()
    model = REINFORCE(env, discount_rate=0.99, callback=test_model)
    model.learn(2 ** -9, 2 ** -6)
