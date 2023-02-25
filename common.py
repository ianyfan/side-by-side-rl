from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Any, NewType

import gymnasium
import torch

# constants
DEFAULT_HIDDEN_SIZE = 256
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_ACTIVATION = torch.nn.ReLU

# types


@dataclass
class Algorithm(ABC):
    """Abstract base class for the learning algorithms.

    Contains common attributes and methods used by most algorithms.

    Parameters
    ----------
    env : gymnasium.Env
        The environment to act in.
    callback : Callable[[Algorithm], None], optional
        Callback that is called during training after each episode with
        the algorithm instance as the sole parameter.

    Attributes
    ----------
    policy : Callable[[torch.Tensor], torch.Tensor]
        The policy that is used for acting, generally defined by
        `self.initialize_networks`. Usually a `torch.nn.Module`. Takes
        the current state and returns the action to take.
    timestep : int
        The number of training steps that have been taken. Not
        necessarily equal to the number of policy iteration steps that
        have been taken.
    _last_observation : torch.Tensor or None
        The last training observation that has been observed, if in the
        middle of an episode, otherwise `None`.

    """
    env: gymnasium.Env

    callback: Callable[[Algorithm], None] | None = field(
        default=None,
        kw_only=True
    )

    policy: Callable[[torch.Tensor], torch.Tensor] = field(init=False)
    timestep: int = field(default=0, init=False)
    _last_observation: torch.Tensor | None = field(default=None, init=False)

    @abstractmethod
    def initialize_networks(
        self
    ) -> torch.nn.Module | tuple[torch.nn.Module, ...]:
        """Initialize the network(s) used by the learning algorithm."""

    def env_reset(self, *, env: gymnasium.Env | None = None) -> torch.Tensor:
        """Reset the envirnoment and return the initial observation.

        Returns the initial observation as a PyTorch tensor.

        """
        if env is None:
            env = self.env
        return torch.as_tensor(env.reset()[0])

    def env_step(
        self,
        action: torch.Tensor,
        *,
        env: gymnasium.Env | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, bool]:
        """Take a step in the environment and process output.

        By default, the self environment is used, but an external
        environment can be specified to act in that environment instead.

        Pre-processing:
        - increment the timestep counter (if using instance environment)
        - clip the action, if needed
        - transform the action from a PyTorch tensor to a NumPy array

        Post-processing:
        - transform the next observation and reward from NumPy arrays to
          PyTorch tensors
        - merge `terminated` and `truncated` outputs into a single
          `done` output

        Parameters
        ----------
        action : torch.Tensor
            The action to take.
        env : gymnasium.Env, optional
            The environment to act in. By default, the self environment
            is used.

        Returns
        -------
        torch.Tensor
            The next observation.
        torch.Tensor
            The reward.
        bool
            Whether the environment is done.

        """

        if env is None:
            env = self.env

        if env is self.env:
            self.timestep += 1

        clipped_action = action.numpy()
        if isinstance(env.action_space, gymnasium.spaces.Box):
            clipped_action = clipped_action.clip(env.action_space.low,
                                                 env.action_space.high)

        (
            next_observation,
            reward,
            terminated,
            truncated,
            info
        ) = env.step(clipped_action)

        return (torch.as_tensor(next_observation),
                torch.as_tensor(reward),
                terminated or truncated)

    def generate_rollout(self,
                         timesteps: int = 0,
                         *,
                         episodes: int = 0,
                         env: gymnasium.Env | None = None) -> Rollout:
        """Generate a rollout.

        Both the number of timesteps or the number of episodes can be
        specified, and the rollout runs until whichever is longer; by
        default, both are 0.

        By default, the self environment is used, but an external
        environment can be specified to act in that environment instead.

        Parameters
        ----------
        timesteps : int
            The (minimum) number of time steps to run for.
        episodes : int
            The (minimum) number of episodes to run for.
        env : gymnasium.Env, optional
            The environment to act in. By default, the self environment
            is used.

        Returns
        -------
        Rollout
            The generated rollout.

        """

        using_self_env = env is None
        if using_self_env:
            env = self.env

        # initialize rollout
        rollout = Rollout()

        # get initial state
        if using_self_env and self._last_observation is not None:
            observation = self._last_observation
        else:
            observation = None

        while timesteps > 0 or episodes > 0:
            # reset environment if needed
            if observation is None:
                observation = self.env_reset(env=env)

            # act
            with torch.no_grad():
                action = self.policy(observation)
            next_observation, reward, done = self.env_step(action, env=env)

            # add to rollout
            rollout.add(observation, action, reward, next_observation, done)

            # update variables
            timesteps -= 1
            if done:
                episodes -= 1
                observation = None
            else:
                observation = next_observation

            if using_self_env:
                self._last_observation = observation

        return rollout

    @abstractmethod
    def learn(self, *args: Any) -> None:
        """Train the policy."""


# PyTorch Modules

def MLP(
    input_size: int,
    output_size: int,
    hidden_size: int = DEFAULT_HIDDEN_SIZE,
    hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
    activation: type = DEFAULT_ACTIVATION,
    output_activation: type | None = None
) -> torch.nn.Module:
    """A multilayer perceptron."""
    assert hidden_layers >= 0

    layer_sizes = (input_size, *[hidden_size]*hidden_layers, output_size)

    layers = []
    for in_size, out_size in zip(layer_sizes, layer_sizes[1:]):
        layers += [torch.nn.Linear(in_size, out_size), activation()]
    del layers[-1]

    if output_activation is not None:
        layers.append(output_activation())

    return torch.nn.Sequential(*layers)


class MLPActorCritic(torch.nn.Module):
    """An MLP-based actor-critic policy with shared architecture.

    The architecture is an MLP for feature extraction, followed by a
    single linear layer each for the actor and the critic.
    The critic can be disabled by passing `critic=False`.

    The input and output size of the actor is inferred by the input
    parameters `observation_space` and `action_space` respectively.
    The output size of the critic is always 1.

    Currently, the only supported action spaces are `Discrete` and
    `Box`. This is generally used for stochastic, on-policy algorithms,
    so if the action space is `Box`, then the output distribution is a
    normal distribution.

    """

    def __init__(self,
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
                 activation: type = DEFAULT_ACTIVATION,
                 critic: bool = True,
                 deterministic: bool = False,
                 tanh_output: bool = False) -> None:
        super().__init__()

        assert len(observation_space.shape) == 1
        input_size = observation_space.shape[0]

        self.action_space = action_space
        self.deterministic = deterministic
        if isinstance(action_space, gymnasium.spaces.Box):
            if deterministic:
                output_size = math.prod(action_space.shape)
            else:
                output_size = 2 * math.prod(action_space.shape)
        elif isinstance(action_space, gymnasium.spaces.Discrete):
            assert not tanh_output
            output_size = action_space.n
        else:
            raise NotImplementedError

        assert hidden_layers > 0
        self.features_extractor = MLP(input_size=input_size,
                                      output_size=hidden_size,
                                      hidden_size=hidden_size,
                                      hidden_layers=hidden_layers - 1,
                                      activation=activation,
                                      output_activation=activation)
        self.actor = torch.nn.Linear(hidden_size, output_size)
        self.critic = torch.nn.Linear(hidden_size, 1) if critic else None

        self.tanh_output = tanh_output

    def distribution(
        self,
        observation: torch.Tensor
    ) -> torch.distributions.Distribution:
        if self.deterministic:
            raise AttributeError('This policy is deterministic.')

        dist_params = self.actor(self.features_extractor(observation))
        if isinstance(self.action_space, gymnasium.spaces.Box):
            mean, untransformed_std_dev = (
                dist_params
                .reshape(-1, 2, *self.action_space.shape)
                .transpose(0, 1)  # swap batch dim with params dim
            )
            return torch.distributions.Independent(
                torch.distributions.Normal(
                    mean,
                    torch.nn.functional.softplus(untransformed_std_dev)
                ),
                mean.ndim - 1
            )
        elif isinstance(self.action_space, gymnasium.spaces.Discrete):
            return torch.distributions.Categorical(logits=dist_params)
        else:
            raise NotImplementedError

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        if self.deterministic:
            action = self.actor(self.features_extractor(observation))
            if isinstance(self.action_space, gymnasium.spaces.Box):
                action = action.reshape(*action.shape[:-1],
                                        *self.action_space.shape)
        else:
            action = self.distribution(observation).sample().squeeze(0)

        if self.tanh_output:
            if isinstance(self.action_space, gymnasium.spaces.Box):
                action = (torch.tanh(action) + 1) / 2  # scale to 0-1

                low = torch.as_tensor(self.action_space.low)
                high = torch.as_tensor(self.action_space.high)
                action = action * (high - low) + low

        return action

    def value(self, observation: torch.Tensor) -> torch.Tensor:
        if self.critic is not None:
            features = self.features_extractor(observation)
            return self.critic(features).squeeze(-1)
        else:
            raise AttributeError('This policy has no critic.')


def MLPActor(*args, **kwargs) -> MLPActorCritic:
    """Make an MLP-based policy with just an actor (no critic)."""
    return MLPActorCritic(*args, **kwargs, critic=False)


def MLPDeterministicActor(*args, **kwargs) -> MLPActorCritic:
    """Make a deterministic MLP-based actor (no critic)."""
    return MLPActorCritic(*args, **kwargs, critic=False, deterministic=True)


class MLPCritic(torch.nn.Module):
    """An MLP-based critic."""

    def __init__(self,
                 observation_space: gymnasium.spaces.Space,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
                 activation: type = DEFAULT_ACTIVATION) -> None:
        super().__init__()
        assert len(observation_space.shape) == 1
        input_size = observation_space.shape[0]

        self.critic = MLP(input_size=input_size,
                          output_size=1,
                          hidden_size=hidden_size,
                          hidden_layers=hidden_layers,
                          activation=activation)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state).squeeze(-1)


class MLPActionCritic(torch.nn.Module):
    """An MLP-based critic that evalates a state-action."""

    def __init__(self,
                 observation_space: gymnasium.spaces.Space,
                 action_space: gymnasium.spaces.Space,
                 hidden_size: int = DEFAULT_HIDDEN_SIZE,
                 hidden_layers: int = DEFAULT_HIDDEN_LAYERS,
                 activation: type = DEFAULT_ACTIVATION) -> None:
        super().__init__()

        assert len(observation_space.shape) == 1
        assert isinstance(action_space, gymnasium.spaces.Box)
        input_size = observation_space.shape[0] + math.prod(action_space.shape)

        self.critic = MLP(input_size=input_size,
                          output_size=1,
                          hidden_size=hidden_size,
                          hidden_layers=hidden_layers,
                          activation=activation)

    def forward(self,
                state: torch.Tensor,
                action: torch.Tensor) -> torch.Tensor:
        action = action.flatten(state.ndim - 1)
        return self.critic(torch.cat((state, action), dim=-1)).squeeze(-1)


# Transition storage, i.e. replay buffers and rollouts
RawTransition = NewType(
    'RawTransition',
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, bool]
)
Transition = NewType(
    'Transition',
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
)


@dataclass
class ReplayData:
    observations: torch.Tensor  # AKA self.states
    actions: torch.Tensor
    rewards: torch.Tensor
    next_observations: torch.Tensor  # AKA self.next_states
    dones: torch.Tensor

    def __init__(self, items: list[RawTransition]) -> None:
        obs, actions, rewards, next_obs, dones = zip(*items)
        self.observations = torch.stack(obs)
        self.actions = torch.stack(actions)
        self.rewards = torch.as_tensor(rewards, dtype=self.observations.dtype)
        self.next_observations = torch.stack(next_obs)
        self.dones = torch.as_tensor(dones, dtype=self.observations.dtype)

    @property
    def states(self) -> torch.Tensor:
        return self.observations

    @property
    def next_states(self) -> torch.Tensor:
        return self.next_observations


@dataclass
class ReplayBuffer:
    capacity: int
    items: list[RawTransition] = field(default_factory=list, init=False)

    def store(self, item: RawTransition) -> None:
        self.items.append(item)
        if len(self.items) > self.capacity:
            del self.items[0]

    def sample(self, batch_size: int) -> ReplayData:
        indices = torch.randint(len(self.items), size=(batch_size,))
        return ReplayData([self.items[i] for i in indices])


@dataclass
class Rollout:
    observations: torch.Tensor = field(default_factory=torch.Tensor)
    actions: torch.Tensor = field(default_factory=torch.Tensor)
    rewards: torch.Tensor = field(default_factory=torch.Tensor)
    next_observations: torch.Tensor = field(default_factory=torch.Tensor)
    dones: torch.Tensor = field(default_factory=torch.Tensor)

    def __len__(self) -> int:
        return len(self.observations)

    def __getitem__(self, index: int | slice) -> Transition:
        return (self.observations[index],
                self.actions[index],
                self.rewards[index],
                self.next_observations[index],
                self.dones[index])

    @staticmethod
    def _append(sequence: torch.Tensor, value: torch.Tensor) -> None:
        """Append a value to a tensor, as if it were mutable."""
        clone = sequence.clone()
        torch.cat((clone, value.unsqueeze(0)), out=sequence.resize_(0))

    def add(self,
            observation: torch.Tensor,
            action: torch.Tensor,
            reward: torch.Tensor,
            next_observation: torch.Tensor | None = None,
            done: bool | None = None) -> None:
        self._append(self.observations, observation)
        self._append(self.actions, action)
        self._append(self.rewards, reward)
        if next_observation is not None:
            self._append(self.next_observations, next_observation)
        if done is not None:
            self._append(self.dones,
                         torch.as_tensor(done, dtype=self.dones.dtype))
