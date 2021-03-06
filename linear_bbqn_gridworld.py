# -*- coding: utf-8 -*-

import gym
import gym_gridworld

import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
from itertools import count
from copy import deepcopy
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from VariationalLinearLayer import LinearLayer


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

    def shuffle(self):
        random.shuffle(self.memory)

    def state_action_counts(self):
        freqs = defaultdict(lambda: defaultdict(int))
        for transition in self.memory:
            state = transition.state[0].numpy()
            state = np.argmax(state)
            action = transition.action[0,0]
            freqs[state][action] += 1
        return freqs

    def __len__(self):
        return len(self.memory)


class BBQN():
    def __init__(self, num_features, num_actions, rho, bias=True):
        self.D_in = num_features
        self.D_out = num_actions
        self.bias = bias
        self.rho = rho

        self.head = LinearLayer(num_features, num_actions, rho, bias=self.bias)
        self.layers = [self.head]

    def make_target(self):
        target = BBQN(self.D_in, self.D_out, self.rho, bias=self.bias)
        target.head = self.head.make_target()
        return target

    def parameters(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.parameters()
        return to_ret

    def get_mu_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_mu_l()
        return to_ret

    def get_rho_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_rho_l()
        return to_ret

    def get_sigma_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_sigma_l()
        return to_ret

    #sample each layer and concatenate the result into a list of variables
    #if set mean  is true, sample is just the mean of each distribution
    def sample(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.sample()
        return to_ret

    def forward(self, x, mean_only):
        return self.head.forward(x, mean_only)
  
    def __call__(self, x, mean_only=False):
        return self.forward(x, mean_only)

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

steps_done = 0
def select_action(model, state):
    var = Variable(state, volatile=True).type(FloatTensor) 
    actions = model(var)
    best = actions.data.max(1)[1]
    return LongTensor([best[0]]).view(1, 1)

def get_Q(target, state):
    var = Variable(state, volatile=True).type(FloatTensor) 
    return target(var, mean_only=True).data

def Q_values(env, target):
    n = env.state_size()
    m = int(n ** 0.5)
    states = np.identity(n)
    Q = torch.zeros(n, env.num_actions())
    for i, row in enumerate(states):
        state = Tensor(row).unsqueeze(0)
        Q[i] = get_Q(target, state)[0]
    return Q

def median_Q(env, target):
    Q = Q_values(env, target)
    return torch.median(Q)

def Q_dump(env, target):
    n = env.state_size()
    m = int(n ** 0.5)
    Q = Q_values(env, target)
    for i, row in enumerate(Q.t()):
        print "Action {}".format(i)
        print row.contiguous().view(m, m)
    
def simulate(RHO_P, STD_DEV_P, BATCH_SIZE, GAMMA, TARGET_RESET_PERIOD, SAMPLE_PERIOD, NUM_EPISODES,
                TRAIN_IN_EPOCHS, NUM_TARGET_RESET, NUM_EPOCHS):
    env = gym.make('gym_onehotgrid-v0').unwrapped
    model = BBQN(env.state_size(), env.num_actions(), RHO_P, bias=False)
    target = model.make_target()
    optimizer = optim.Adam(model.parameters())
    memory = ReplayMemory(2**15)

    last_sync = [0]
    loss_comp_dict = defaultdict(list)
    loss_component_names = ['Log sigma', 'W over prior', 'W over posterior', 'Q error']
    sigma_average_dict = defaultdict(list)
    components = ['W']
    loss_average_l = []
    median_q_l = []
    print RHO_P, STD_DEV_P

    def optimize_model(w_sample, model, target):
        if len(memory) < BATCH_SIZE:
            return model, target
        
        if last_sync[0] == TARGET_RESET_PERIOD:
            target = model.make_target()
            median_q_l.append(median_Q(env, target))
            print "Target reset"
            # Q_dump(env, target)
            last_sync[0] = 0
        last_sync[0] += 1

        def loss_of_batch(batch):
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
            state_action_values = model(state_batch).gather(1, action_batch).view(-1)

            # Compute V(s_{t+1}) for all next states.
            next_state_values = Variable(torch.zeros(BATCH_SIZE).type(Tensor))
            next_state_values[non_final_mask] = target(non_final_next_states, mean_only=True).max(1)[0]
            # Now, we don't want to mess up the loss with a volatile flag, so let's
            # clear it. After this, we'll just end up with a Variable that has
            # requires_grad=False
            next_state_values.volatile = False
            # Compute the expected Q values
            expected_state_action_values = (next_state_values * GAMMA) + reward_batch

            loss_components = [0.0, 0.0, 0.0, 0.0]
            # Compute l2 loss
            loss_components[3] = (state_action_values - expected_state_action_values).pow(2).sum()

            #Now add log(q(w|theta)) - log(p(w)) terms
            mu_l = model.get_mu_l()
            sigma_l = model.get_sigma_l()
            for i in range(len(w_sample)):
                w = w_sample[i]
                mu = mu_l[i]
                sigma = sigma_l[i]
                loss_components[0] -= torch.log(sigma).sum()
                loss_components[1] += (w.pow(2)).sum() / (2.0 * (STD_DEV_P ** 2))
                loss_components[2] -= ((w - mu).pow(2) / (2.0 * sigma.pow(2))).sum()

            for i in range(len(loss_components)-1):
                loss_components[i] /= M

            for i in range(len(loss_components)):
                loss_comp_dict[loss_component_names[i]].append(loss_components[i].data[0])
                    
            loss = sum(loss_components)
            
            return loss

        def gradient_step(loss):
            # Optimize the model
            optimizer.zero_grad()
            loss.backward()
            # nn.utils.clip_grad_norm(model.parameters(), 1.0)
            optimizer.step()

        if TRAIN_IN_EPOCHS:
            M = len(memory) / BATCH_SIZE
            for target_iter in range(NUM_TARGET_RESET):
                target = model.make_target()
                memory.shuffle()
                for epoch in range(NUM_EPOCHS):           
                    for minibatch in range(M):
                        start_idx = minibatch * BATCH_SIZE
                        end_idx = start_idx + BATCH_SIZE
                        transitions = memory.memory[start_idx:end_idx]
                        w_sample = model.sample()
                        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
                        # detailed explanation).
                        batch = Transition(*zip(*transitions))

                        loss = loss_of_batch(batch)
                        loss_average_l.append(loss.data[0])
                        gradient_step(loss)
        else:
            transitions = memory.sample(BATCH_SIZE)
            M = 1

            # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
            # detailed explanation).
            batch = Transition(*zip(*transitions))

            loss = loss_of_batch(batch)
            loss_average_l.append(loss.data[0])
            gradient_step(loss)

        return model, target
    
    score_list = []
    median_q_l = []
    for i_episode in range(NUM_EPISODES):
        # Initialize the environment and state
        env.reset()
       
        state = Tensor(env.get_state()).unsqueeze(0)
        iters = 0
        score = 0
        while iters < 50:
            do_update = False
            if iters % SAMPLE_PERIOD == 0:
                w_sample = model.sample()
                do_update = not TRAIN_IN_EPOCHS
            iters += 1
            
            # Select and perform an action
            action = select_action(model, state)
            next_state, reward, done, _ = env.step(action[0, 0])
            next_state = Tensor(next_state).unsqueeze(0)
            score += reward
            reward = Tensor([reward])

            if done:
                next_state = None

            # Store the transition in memory
            memory.push(state, action, next_state, reward)

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            if do_update:
                model, target = optimize_model(w_sample, model, target)
            if done:
                break

        median_q_l.append(median_Q(env, target))
        score_list.append(score)
        for idx, sigma in enumerate(model.get_sigma_l()):
            average = sigma.mean().data[0]
            sigma_average_dict[components[idx]].append(average)
        if i_episode % 100 == 0:
            print "Episode: {}\tscore: {}".format(i_episode, score)
        if TRAIN_IN_EPOCHS and i_episode % TRAIN_IN_EPOCHS_PERIOD == 0:
            model, target = optimize_model(w_sample, model, target)

    print memory.state_action_counts()
    Q_dump(env, target)
    print model.get_mu_l()
    print model.get_sigma_l()
    return sigma_average_dict['W'], loss_average_l, median_q_l, score_list

handles = []
loss_average_l = []
median_q_l = []
score_l = []
# rho_l = [-3.0, -2.0, -1.0, 0., 1., 2.0]
rho_l = [0.]
for rho in rho_l:
    ### Hyperparameters
    RHO_P = rho
    STD_DEV_P = math.log1p(math.exp(RHO_P))

    BATCH_SIZE = 256
    GAMMA = 0.999
    TARGET_RESET_PERIOD = 5000
    SAMPLE_PERIOD = 5
    NUM_EPISODES = 10000

    TRAIN_IN_EPOCHS = True
    NUM_TARGET_RESET = 2
    NUM_EPOCHS = 2
    TRAIN_IN_EPOCHS_PERIOD = 50
    ### 

    sigma_average, loss_average, median_q, score = simulate(RHO_P, STD_DEV_P, BATCH_SIZE, GAMMA, TARGET_RESET_PERIOD,\
            SAMPLE_PERIOD, NUM_EPISODES, TRAIN_IN_EPOCHS, NUM_TARGET_RESET, NUM_EPOCHS)
    handle, = plt.semilogy(sigma_average, label="RHO={}".format(rho))
    handles.append(handle)
    loss_average_l.append(loss_average)
    median_q_l.append(median_q)
    score_l.append(score)
plt.legend(handles=handles)
plt.xlabel("Number of episodes")
plt.ylabel("Mean of uncertainty in each layer")
plt.title("Uncertainty versus episodes")
plt.savefig("RHO_decay_of_average_sigma.png")
plt.show()    

handles = []
for i, values in enumerate(median_q_l):
    handle, = plt.plot(values, label="RHO={}".format(rho_l[i]))
    handles.append(handle)
plt.legend(handles=handles)
plt.xlabel("Number of target updates")
plt.ylabel("Median Q value")
plt.title("Growth of median Q value with target updates")
plt.savefig("RHO_median_q_value.png")
plt.show()

handles = []
N = 100
for i, values in enumerate(loss_average_l):
    averages = np.convolve(values, np.ones((N,))/N, mode='valid')
    handle, = plt.plot(averages, label="RHO={}".format(rho_l[i]))
    handles.append(handle)
plt.legend(handles=handles)
plt.xlabel("Number of updates")
plt.ylabel("Smoothed Loss")
plt.title("Loss versus number of updates")
plt.savefig("RHO_loss.png")
plt.show()

handles = []
N = 100
for i, values in enumerate(score_l):
    averages = np.convolve(values, np.ones((N,))/N, mode='valid')
    handle, = plt.plot(averages, label="RHO={}".format(rho_l[i]))
    handles.append(handle)
plt.legend(handles=handles)
plt.xlabel("Number of episodes")
plt.ylabel("Score (Max 7)")
plt.title("Score versus number of episodes")
plt.savefig("RHO_score.png")
plt.show()
