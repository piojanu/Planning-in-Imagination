import time

import torch
import torch.nn.functional as F
import numpy as np
from matplotlib import pyplot as plt
import sys


def replay_buffer(buffer_dir="buffer.npz", interval=0.001):
    npzfile = np.load(buffer_dir)
    for state, action in zip(npzfile["states"], npzfile["actions"]):
        plt.imshow(state)
        plt.show(block=False)
        plt.pause(interval)


def adjust_learning_rate(optimizer, init_lr, gamma, global_step, decay_interval):
    num_decays = global_step//decay_interval
    lr = init_lr * gamma**num_decays
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def eval_in_batches(model, dataset, reward_loss_factor=0.01, eval_size=1000):
    if eval_size > dataset.num_samples:
        eval_size = dataset.num_samples
    num_batches = dataset.num_samples//eval_size
    loss_accumulated = 0

    for i in range(num_batches):
        states_batch, next_states_batch, actions_batch, rewards_batch = dataset.get_next_batch(eval_size)
        states_pred, rewards_pred = model(torch.Tensor(states_batch), torch.Tensor(actions_batch))
        loss = F.mse_loss(states_pred, torch.Tensor(next_states_batch))
        rewards_loss = reward_loss_factor * F.mse_loss(rewards_pred, torch.Tensor(rewards_batch))

        loss_accumulated += loss.item() + rewards_loss.item()

    return loss_accumulated/num_batches


def test_model(model, dataset, interval=0.003):
    fig = plt.figure(figsize=(1, 3))
    model = torch.load("{}.model".format(model.name))
    for i in range(dataset.num_samples):
        states, next_states, actions, rewards = dataset.get_next_batch(1)
        state_pred, reward_pred = model(torch.Tensor(states), torch.Tensor(actions))

        fig.add_subplot(1, 3, 1)
        plt.suptitle("Step: {}, reward_pred: {:.5f}".format(i, reward_pred.detach().cpu().numpy()[0][0]))
        plt.imshow(states[0][-1].astype(np.float32))
        fig.add_subplot(1, 3, 2)
        plt.imshow(state_pred.detach().cpu().numpy().reshape(80, 80))
        fig.add_subplot(1, 3, 3)
        plt.imshow(next_states[0][0].astype(np.float32))

        plt.show(block=False)

        plt.pause(interval)
