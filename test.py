import datetime
import importlib
import statistics
import sys

import gymnasium
import torch


def demonstrate(model):
    test_env = gymnasium.make(ENV, render_mode='human', **ENV_ARGS)
    test_env.reset(seed=SEED)
    test_env.action_space.seed(seed=SEED)

    while True:
        rollout = model.generate_rollout(episodes=1, env=test_env)
        print(float(sum(rollout.rewards)))


def now():
    return datetime.datetime.now().replace(microsecond=0)


episode = 0


def callback(model):
    global episode
    episode += 1
    if episode % every:
        return

    runs = 10
    returns = []

    test_env = gymnasium.make(ENV, **ENV_ARGS)
    test_env.reset(seed=SEED)
    test_env.action_space.seed(seed=SEED)
    with torch.random.fork_rng():
        for _ in range(runs):
            rollout = model.generate_rollout(episodes=1, env=test_env)
            returns.append(float(sum(rollout.rewards)))

    mean = statistics.fmean(returns)
    stddev = statistics.stdev(returns)

    print(f'{now()} - Episode {episode} ({model.timestep} timesteps): '
          f'return = {mean:g}±{stddev:g}', end='')

    if all(ret >= solved_threshold for ret in returns):
        with torch.random.fork_rng():
            for _ in range(2 * runs):
                rollout = model.generate_rollout(episodes=1, env=test_env)
                ret = float(sum(rollout.rewards))
                if ret < solved_threshold:
                    print(f' >{ret:g}')
                    break
            else:
                print()
                demonstrate(model)
    else:
        print(f' >{min(returns):g}')


ENV = sys.argv[2]
ENV_ARGS = {}
SEED = 0
torch.manual_seed(SEED)

train_env = gymnasium.make(ENV, **ENV_ARGS)
train_env.reset(seed=SEED)
train_env.action_space.seed(seed=SEED)

alg = sys.argv[1]
solved_threshold = int(sys.argv[3])
try:
    every = int(sys.argv[4])
except (IndexError, ValueError):
    every = 1

print('Alg:', alg)
print('Env:', ENV)
print('Solved threshold:', solved_threshold)
print('Every:', every)
print(f'Started {now()}')
importlib.import_module(f'{alg}.{alg}').test(train_env, callback)
