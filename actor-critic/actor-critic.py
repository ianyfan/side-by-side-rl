from collections.abc import Callable
from dataclasses import dataclass, field

from gymnasium import Env
import torch

from common import Algorithm, MLPCritic, MLPPolicy


@dataclass
class ActorCritic(Algorithm):
    discount_rate: float

    critic: torch.nn.Module = field(init=False)

    def initialize_networks(self) -> tuple[MLPPolicy, MLPCritic]:
        return (MLPPolicy(self.env.observation_space, self.env.action_space),
                MLPCritic(self.env.observation_space))

    # LINE 0
    def learn(
        # LINE 1-2
        self,
        # LINE 3
        policy_step_size: float,
        critic_step_size: float,
        # LINE
        epochs: int
    ) -> None:
        # LINE 4
        self.policy, self.critic = self.initialize_networks()
        # LINE
        policy_optimizer = torch.optim.SGD(
            self.policy.parameters(),
            lr=policy_step_size,
            maximize=True
        )
        critic_optimizer = torch.optim.SGD(
            self.critic.parameters(),
            lr=critic_step_size
        )
        # LINE 5
        for _ in range(epochs):
            # LINE 6
            state = torch.as_tensor(self.env.reset()[0])
            # LINE 7
            discount = 1
            # LINE 8
            done = False
            while not done:
                # LINE 9
                action = self.policy(state)
                # LINE 10
                next_state, reward, done = self.step(action)
                # LINE 11-12
                if done:
                    next_value = 0
                else:
                    with torch.no_grad():
                        next_value = self.critic(next_state)

                temporal_difference = (
                    reward
                    + self.discount_rate * next_value
                    - self.critic(state)
                )
                # LINE 13
                value_loss = temporal_difference ** 2

                critic_optimizer.zero_grad()
                value_loss.backward()
                critic_optimizer.step()
                # LINE 14
                policy_loss = (
                    discount
                    * temporal_difference.detach()
                    * self.policy.distribution(state).log_prob(action)
                )

                policy_optimizer.zero_grad()
                policy_loss.backward()
                policy_optimizer.step()
                # LINE 15
                discount *= self.discount_rate
                # LINE 16
                state = next_state
                # END

            if self.callback is not None:
                self.callback(self)


def test(env: Env, callback: Callable[[Algorithm], None] | None) -> None:
    model = ActorCritic(env, discount_rate=1, callback=callback)
    model.learn(policy_step_size=1e-6,
                critic_step_size=1e-5,
                epochs=1000000)
