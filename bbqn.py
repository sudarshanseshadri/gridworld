# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) tutorial
=====================================
**Author**: `Adam Paszke <https://github.com/apaszke>`_


This tutorial shows how to use PyTorch to train a Deep Q Learning (DQN) agent
on the CartPole-v0 task from the `OpenAI Gym <https://gym.openai.com/>`__.


**Packages**

First, let's import needed packages. Firstly, we need
`gym <https://gym.openai.com/docs>`__ for the environment
(Install using `pip install gym`).
We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)
-  utilities for vision tasks (``torchvision`` - `a separate
   package <https://github.com/pytorch/vision>`__).

"""
import sys
from gridworld import GridWorld
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from copy import deepcopy

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

env = GridWorld(4, (0,0), {(1, 1):1, (3, 3):10})

# if gpu is to be used
use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
ByteTensor = torch.cuda.ByteTensor if use_cuda else torch.ByteTensor
Tensor = FloatTensor


######################################################################
# Replay Memory
# -------------
#
# We'll be using experience replay memory for training our DQN. It stores
# the transitions that the agent observes, allowing us to reuse this data
# later. By sampling from it randomly, the transitions that build up a
# batch are decorrelated. It has been shown that this greatly stabilizes
# and improves the DQN training procedure.
#
# For this, we're going to need two classses:
#
# -  ``Transition`` - a named tuple representing a single transition in
#    our environment
# -  ``ReplayMemory`` - a cyclic buffer of bounded size that holds the
#    transitions observed recently. It also implements a ``.sample()``
#    method for selecting a random batch of transitions for training.
#

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class LinearLayer():
    def __init__(self, d_in, d_out, dtype=FloatTensor):
        self.d_in = d_in
        self.d_out = d_out
        self.W_mu = Variable(torch.Tensor(d_out, d_in).type(dtype), requires_grad=True)
        # self.W_rho = Variable(torch.Tensor(d_in, d_out).type(dtype), requires_grad=True)
        self.b_mu = Variable(torch.Tensor(d_out).type(dtype), requires_grad=True)
        # self.b_rho = Variable(torch.Tensor(d_out).type(dtype), requires_grad=True)
        # self.W_sample = None
        # self.b_sample = None
        # self.components = [self.W_mu, self.W_rho, self.b_mu, self.b_rho]
        self.components = [self.W_mu, self.b_mu]

    def parameters(self):
        return self.components

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.W_mu.data.uniform_(-stdv, stdv)
        self.b_mu.data.uniform_(-stdv, stdv)

    #Use the current distribution parameters over W and b to sample W and b.
    #Saves the sample Variable in self.W_sample and self.b_sample, and returns the samples
    # def sample(self):

        #given rho, calculate standard deviation (sigma)
        # def calc_sigma(rho):
            # return torch.log(1.0 + torch.exp(rho))
        
        # W_sigma = calc_sigma(self.W_rho)
        # b_sigma = calc_sigma(self.b_rho)
        # W_sample = self.W_mu + Variable(torch.randn(self.d_in, self.d_out), requires_grad=False) * W_sigma
        # b_sample = self.b_mu + Variable(torch.randn(self.d_out), requires_grad=False) * b_sigma
        
        ##TODO: change this to sample
        # # self.W_sample = W_sample
        # # self.b_sample = b_sample
        # self.W_sample = self.W_mu
        # self.b_sample = self.b_mu
        # return [self.W_sample, self.b_sample]

    #calculate x*W_sample + b_sample, and return the Variable
    def forward(self, x):
        # if self.W_sample is None or self.b_sample is None:
        #     raise Exception("Must sample W and b before calling forward")
        # return torch.matmul(x, self.W_sample) + self.b_sample
        return F.linear(x, self.W_mu, self.b_mu)

    # def zero_grad(self):
    #     for var in self.components:
    #         if var.grad is not None:
    #             var.grad.data.zero_()

    # def update(self, lr):
    #     for var in self.components:
    #         if var.grad is not None:
    #             var.data -= lr * var.grad.data
    #         var.data.clamp_(-1, 1)

class DQN():
    def __init__(self):
        D_in, H, D_out = 2, 8, 4

        self.h1 = LinearLayer(D_in, H)
        self.h2 = LinearLayer(H, H)
        self.head = LinearLayer(H, D_out)
        self.components = [self.h1, self.h2, self.head]

    def parameters(self):
        to_ret = []
        for layer in self.components:
            to_ret += layer.parameters()
        return to_ret

    #sample each layer and concatenate the result into a list of variables
    # def sample(self):
    #     to_ret = []
    #     for layer in self.components:
    #         to_ret += layer.sample()
    #     return to_ret

    def forward(self, x):
        x = F.softmax(self.h1.forward(x))
        x = F.softmax(self.h2.forward(x))
        return self.head.forward(x)

    def __call__(self, x):
        return self.forward(x)

    # def zero_grad(self):
    #      for layer in self.components:
    #         layer.zero_grad()
                    
    # def update(self, lr):
    #     for layer in self.components:
    #         layer.update(lr)

######################################################################
# Training
# --------
#
# Hyperparameters and utilities
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# This cell instantiates our model and its optimizer, and defines some
# utilities:
#
# -  ``Variable`` - this is a simple wrapper around
#    ``torch.autograd.Variable`` that will automatically send the data to
#    the GPU every time we construct a Variable.
# -  ``select_action`` - will select an action accordingly to an epsilon
#    greedy policy. Simply put, we'll sometimes use our model for choosing
#    the action, and sometimes we'll just sample one uniformly. The
#    probability of choosing a random action will start at ``EPS_START``
#    and will decay exponentially towards ``EPS_END``. ``EPS_DECAY``
#    controls the rate of the decay.
# -  ``plot_durations`` - a helper for plotting the durations of episodes,
#    along with an average over the last 100 episodes (the measure used in
#    the official evaluations). The plot will be underneath the cell
#    containing the main training loop, and will update after every
#    episode.
#

BATCH_SIZE = 64
GAMMA = 1.0
EPS_START = 1.0
EPS_END = 0.001
EPS_DECAY = 500

model = DQN()

if use_cuda:
    model.cuda()

optimizer = optim.RMSprop(model.parameters())
memory = ReplayMemory(1000)

steps_done = 0

def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        var = Variable(state, volatile=True).type(FloatTensor) 
        actions = model(var)
        best = actions.data.max(1)[1]
        return LongTensor([best[0]]).view(1, 1)
    else:
        return LongTensor([[random.randrange(4)]])



######################################################################
# Training loop
# ^^^^^^^^^^^^^
#
# Finally, the code for training our model.
#
# Here, you can find an ``optimize_model`` function that performs a
# single step of the optimization. It first samples a batch, concatenates
# all the tensors into a single one, computes :math:`Q(s_t, a_t)` and
# :math:`V(s_{t+1}) = \max_a Q(s_{t+1}, a)`, and combines them into our
# loss. By defition we set :math:`V(s) = 0` if :math:`s` is a terminal
# state.


last_sync = 0


def optimize_model():
    global last_sync
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
    # detailed explanation).
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    non_final_mask = ByteTensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)))

    # We don't want to backprop through the expected action values and volatile
    # will save us on temporarily changing the model parameters'
    # requires_grad to False!
    non_final_next_states = Variable(torch.cat([s for s in batch.next_state
                                                if s is not None]),
                                     volatile=True)
    state_batch = Variable(torch.cat(batch.state))
    action_batch = Variable(torch.cat(batch.action))
    reward_batch = Variable(torch.cat(batch.reward))

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken
    state_action_values = model(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
    next_state_values[non_final_mask] = model(non_final_next_states).max(1)[0]
    # Now, we don't want to mess up the loss with a volatile flag, so let's
    # clear it. After this, we'll just end up with a Variable that has
    # requires_grad=False
    next_state_values.volatile = False
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    for param in model.parameters():
        param.grad.data.clamp_(-1, 1)
    optimizer.step()

######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once. When the episode ends (our model
# fails), we restart the loop.
#
# Below, `num_episodes` is set small. You should download
# the notebook and run lot more epsiodes.


def policy_dump(env):
    size = env.size
    for i in range(size):
        for j in range(size):
            state = Tensor((i,j)).unsqueeze(0)
            action = select_action(state)
            print "State: {}; Action: {}".format(state[0], action[0,0])

total_score_l = []

num_episodes = 2000
for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    
    state = Tensor(env.get_state()).unsqueeze(0)

    score = 0
    for t in xrange(500):
        # Select and perform an action
        action = select_action(state)
        reward, done = env.do_action(action[0, 0])
        score += reward
        reward = Tensor([reward])

        # Observe new state
        if not done:
            next_state = Tensor(env.get_state()).unsqueeze(0)
        else:
            next_state = None

        # Store the transition in memory
        memory.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the target network)
        optimize_model()
        if done:
            if i_episode % 2 == 0:
                print "Iter {}, score: {}".format(i_episode, score)
            total_score_l.append(score)
            break

policy_dump(env)
N = 100
averages = np.convolve(total_score_l, np.ones((N,))/N, mode='valid')
plt.plot(xrange(len(averages)), averages)
plt.show()

