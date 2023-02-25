from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from gymnasium import Env
import torch

from common import Algorithm, MLPActor, MLPCritic


@dataclass
class A2C(Algorithm):
    def initialize_networks(self) -> tuple[MLPActor, MLPCritic]:
        return (MLPActor(self.env.observation_space, self.env.action_space),
                MLPCritic(self.env.observation_space))

    # LINE 0
    def learn(
        self,
        learning_rate: float,
        critic_learning_rate: float,
        rollout_length: int,  # t_max
        discount_rate: float  # γ
    ) -> None:
        # LINE 1-2
        (
            self.policy,
            self.critic
        ) = self.initialize_networks()
        # LINE
        policy_optimizer = torch.optim.RMSprop(
            self.policy.parameters(),
            lr=learning_rate,
            maximize=True
        )
        critic_optimizer = torch.optim.RMSprop(
            self.critic.parameters(),
            lr=critic_learning_rate,
        )
        terminal = True
        # LINE 3
        # not needed for
        # synchronous implementation
        # LINE 4
        while True:
            # LINE 5
            policy_optimizer.zero_grad()
            critic_optimizer.zero_grad()
            # LINE 6-7
            # not needed for
            # synchronous implementation
            # LINE
            states = []
            actions = []
            rewards = []
            # LINE 8
            if terminal:
                state = self.env_reset()
            # LINE 9
            for t in range(rollout_length):
                # LINE 10
                action = self.policy(state)
                # LINE 11
                (
                    next_state,
                    reward,
                    terminal
                ) = self.env_step(action)
                # LINE
                states.append(state)
                actions.append(action)
                rewards.append(reward)
                state = next_state
                # LINE 12
                # t
                # LINE 13
                # self.timestep
                # LINE 14
                if terminal:
                    break
            # LINE 15
            if terminal:
                return_ = 0
            else:
                with torch.no_grad():
                    return_ = self.critic(state).item()
            # LINE 16
            for i in reversed(range(len(states))):
                # LINE 17
                return_ = (
                    rewards[i]
                    + discount_rate * return_
                )
                # LINE
                advantage = (
                    return_
                    - self.critic(states[i])
                )
                # LINE 18
                policy_loss = (
                    self.policy.distribution(
                        states[i]
                    ).log_prob(
                        actions[i]
                    )
                    * advantage.detach()
                )
                policy_loss.backward()
                # LINE 19
                critic_loss = advantage ** 2
                critic_loss.backward()
            # LINE 20
            # LINE 21
            policy_optimizer.step()
            critic_optimizer.step()
            # LINE 22
            # T_max not used here (run forever)
            # END
            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = A2C(env,
                callback=callback)
    model.learn(learning_rate=1e-4,
                critic_learning_rate=1e-3,
                rollout_length=5,
                discount_rate=0.99)
