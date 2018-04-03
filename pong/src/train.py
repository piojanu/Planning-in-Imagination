import torch
import gym
from utils.preprocess import preprocess
from torch.autograd import Variable
from utils.model import PolicyGradient

dtype = torch.FloatTensor if not torch.cuda.is_available() else torch.cuda.FloatTensor

policy = PolicyGradient()


def get_action(observation):
    current_state = preprocess(observation)
    if get_action.prev_state is None:
        get_action.prev_state = current_state
    diff_state = current_state - get_action.prev_state
    get_action.prev_state = current_state
    var_state = Variable(torch.from_numpy(diff_state).type(dtype).unsqueeze(0))
    policy_action = policy.forward(var_state)
    out_action = policy_action.multinomial()
    policy.actions.append(out_action)
    return out_action.data[0, 0] + 1  # Pong specific


env = gym.make("Pong-v0")
observation = env.reset()
env.seed(1)
torch.manual_seed(1)
get_action.prev_state = None  # used in computing the difference frame


import torch.optim as optim

learning_rate = 1e-3
weight_decay = 1e-3

optimizer = optim.RMSprop(
    policy.parameters(),
    lr=learning_rate,
    weight_decay=weight_decay
)
optimizer.zero_grad()


gamma = 0.99


def discount_rewards(rewards):
    current_reward = 0
    out_rewards = []
    for i in reversed(range(len(rewards))):
        if rewards[i] != 0:
            current_reward = 0  # Reset sum between lossing ball
        current_reward = gamma * current_reward + rewards[i]
        out_rewards.insert(0, current_reward)
    return out_rewards


import torch.autograd as autograd
import numpy as np
from datetime import datetime

running_reward = None
rewards = []
reward_sum = 0
batch_size = 1
num_episodes = 0
while True:
    action = get_action(observation)
    observation, reward, done, _ = env.step(action)
    rewards.append(reward)
    reward_sum += reward

    if done:
        num_episodes += 1
        discounted_rewards = discount_rewards(rewards)
        rewards_tensor = dtype(discounted_rewards)
        rewards_tensor = (rewards_tensor - rewards_tensor.mean()) / (rewards_tensor.std() + np.finfo(np.float32).eps)
        for action, reward in zip(policy.actions, rewards_tensor):
            action.reinforce(reward)
           # pass

        rewards = []
        autograd.backward(policy.actions, [None for a in policy.actions])

        if num_episodes % batch_size == 0:
            optimizer.step()
            optimizer.zero_grad()
            print("Updated parameters")

        policy.reset()
        observation = env.reset()
        get_action.prev_state = None

        reward_factor = 1 / num_episodes

        running_reward = reward_sum if running_reward is None else \
            running_reward * (1 - reward_factor) + reward_sum * reward_factor
        print('{:>5} | {} | Episode reward total was {:d}. Running mean: {:.5f}'
              .format(num_episodes, datetime.now().strftime('%H:%M:%S'),
                      int(reward_sum), running_reward))
        if num_episodes % 25 == 0:
            directory = 'models'
            if len(directory) > 0 and directory[-1] == '/':
                directory = directory[0:-1]

            path = "{}/model_rr_{:.3f}_epi_{}.pt".format(
                directory, running_reward, num_episodes)
            torch.save(policy.state_dict(), path)
            print("### Saved model: {} ###".format(path))

        reward_sum = 0
