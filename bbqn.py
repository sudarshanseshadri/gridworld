# -*- coding: utf-8 -*-
"""
Reinforcement Learning (DQN) tutorial
=====================================
**Base Author**: `Adam Paszke <https://github.com/apaszke>`_
**Code apated for BBQN by 'Sudarshan Seshadri'
**BBQN reference: https://arxiv.org/pdf/1608.05081.pdf

**Packages**

First, let's import needed packages. Firstly, we need
`gym <https://gym.openai.com/docs>`__ for the environment
(Install using `pip install gym`).
We'll also use the following from PyTorch:

-  neural networks (``torch.nn``)
-  optimization (``torch.optim``)
-  automatic differentiation (``torch.autograd``)

"""
import sys
from gridworld import GridWorld
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, defaultdict
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

class AllMemory():
    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))

    def clear(self):
        self.memory = []

    def shuffle(self):
        random.shuffle(self.memory)

    def __len__(self):
        return len(self.memory)

class LinearLayer():
    def __init__(self, d_in, d_out, dtype=FloatTensor):
        self.d_in = d_in
        self.d_out = d_out
       
        self.W_mu = Variable(torch.Tensor(d_out, d_in).type(dtype), requires_grad=True)
        self.W_rho = Variable(torch.Tensor(self.W_mu.size()).type(dtype), requires_grad=True)
        self.b_mu = Variable(torch.Tensor(d_out).type(dtype), requires_grad=True)
        self.b_rho = Variable(torch.Tensor(self.b_mu.size()).type(dtype), requires_grad=True)
      
        #sigma_p is a 'hyperparameter' that denotes the initial values for std dev in the parameter distributions
        #it's value is set by self.reset_parameters()
        self.sigma_p_W = None
        self.sigma_p_b = None
        self.reset_parameters() #initializes sigma_p for W and b

        self.W_sample = None
        self.b_sample = None
        self.mu_l = [self.W_mu, self.b_mu]
        self.rho_l = [self.W_rho, self.b_rho]
        self.sigma_p_l = [self.sigma_p_W, self.sigma_p_b]

    # given rho, calculate standard deviation (sigma)
    @staticmethod
    def calc_sigma(rho):
        return torch.log(1.0 + torch.exp(rho))

    def get_mu_l(self):
        return self.mu_l

    def get_rho_l(self):
        return self.rho_l

    def get_sigma_l(self):
        return [LinearLayer.calc_sigma(rho) for rho in self.rho_l]

    def get_sigma_p_l(self):
        return self.sigma_p_l

    def parameters(self):
        return self.get_mu_l() + self.get_rho_l()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.W_mu.size(1)) / 20.0
        stdv = math.expm1(stdv)
        self.W_mu.data.fill_(0)
        self.W_rho.data.fill_(stdv)
        self.sigma_p_W = stdv

        stdv = 1. / math.sqrt(self.b_mu.size(0)) / 20.0
        stdv = math.expm1(stdv)
        self.b_mu.data.fill_(0)
        self.b_rho.data.fill_(stdv)
        self.sigma_p_b = stdv

    def make_target(self):
        target = LinearLayer(self.d_in, self.d_out)
        target.W_mu = Variable(self.W_mu.data.clone(), requires_grad=False)
        target.b_mu = Variable(self.b_mu.data.clone(), requires_grad=False)
        return target

    #Use the current distribution parameters over W and b to sample W and b.
    #Saves the sample Variable in self.W_sample and self.b_sample, and returns the samples
    def sample(self):

        W_sigma = LinearLayer.calc_sigma(self.W_rho)
        b_sigma = LinearLayer.calc_sigma(self.b_rho)
       
        W_sample = self.W_mu + Variable(torch.randn(self.W_mu.size()), requires_grad=True) * W_sigma
        b_sample = self.b_mu + Variable(torch.randn(self.b_mu.size()), requires_grad=True) * b_sigma
        
        self.W_sample = W_sample
        self.b_sample = b_sample

        return [self.W_sample, self.b_sample]

    #calculate x*W_sample + b_sample, and return the Variable
    def forward(self, x):
        if self.W_sample is None or self.b_sample is None:
            raise Exception("Must sample W and b before calling forward")
        return F.linear(x, self.W_sample, self.b_sample)

    #calculate x*W_mu + b_mu
    def forward_mean(self, x):
        return F.linear(x, self.W_mu, self.b_mu)


class BBQN():
    def __init__(self):
        D_in, H, D_out = 2, 4, 4

        self.h1 = LinearLayer(D_in, H)
        self.h2 = LinearLayer(H, H)
        self.head = LinearLayer(H, D_out)
        self.layers = [self.h1, self.h2, self.head]

    def make_target(self):
        target = BBQN()
        target.h1 = self.h1.make_target()
        target.h2 = self.h2.make_target()
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

    def get_sigma_p_l(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.get_sigma_p_l()
        return to_ret

    #sample each layer and concatenate the result into a list of variables
    #if set mean  is true, sample is just the mean of each distribution
    def sample(self):
        to_ret = []
        for layer in self.layers:
            to_ret += layer.sample()
        return to_ret

    def forward(self, x):
        x = F.softmax(self.h1.forward(x))
        x = F.softmax(self.h2.forward(x))
        return self.head.forward(x)

    def forward_mean(self, x):
        x = F.softmax(self.h1.forward_mean(x))
        x = F.softmax(self.h2.forward_mean(x))
        return self.head.forward_mean(x)      

    def __call__(self, x, mean_only=False):
        if mean_only:
            return self.forward_mean(x)
        else:
            return self.forward(x)

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


###Hyperparameters
GAMMA = 1.0
# BATCH_SIZE = 128
NUM_TARGET_RESET = 4
NUM_EPOCHS = 4
NUM_SAMPLES = 4

model = BBQN()

if use_cuda:
    model.cuda()

# optimizer = optim.RMSprop(model.parameters())
optimizer = optim.Adam(model.parameters())


# memory = ReplayMemory(1000)
memory = AllMemory()

# def select_action(state):
#     var = Variable(state, volatile=True).type(FloatTensor) 
#     actions = model(var)
#     best = actions.data.max(1)[1]
#     return LongTensor([best[0]]).view(1, 1)

def select_action(state):
    if 0.5 < random.random():
        return LongTensor([0]).view(1,1)
    else:
        return LongTensor([3]).view(1,1) 

def get_Q(state):
    var = Variable(state, volatile=True).type(FloatTensor) 
    return model(var).data
    

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

loss_list = []
loss_comp_dict = defaultdict(list)
loss_component_names = ['Log sigma', 'W over prior', 'W over posterior', 'Q error']
correlations = []
def optimize_model():
    # if len(memory) < BATCH_SIZE:
    #     return
    average_loss = 0.0
    average_loss_components = [0.0, 0.0, 0.0, 0.0]
    for target_iter in range(NUM_TARGET_RESET):
        target = model.make_target()
        for epoch in range(NUM_EPOCHS):
            # memory.shuffle()
            for sample_iter in range(NUM_SAMPLES):
                w_sample = model.sample()
                loss_components = [0.0, 0.0, 0.0, 0.0]

                # for idx in range(0, len(memory)-BATCH_SIZE, BATCH_SIZE):
                    # transitions = memory.sample(BATCH_SIZE)
                # transitions = memory.memory[idx:idx+BATCH_SIZE]
                transitions = memory.memory
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
                # using MAP to calculate Q(s,a), so just take the mean of parameters
                state_action_values = model(state_batch, mean_only=True).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                next_state_values = Variable(torch.zeros(len(transitions)).type(Tensor))
                

                #DOING POLICY EVAL with 1/2 chance of 0, 1/2 chance of 3
                # next_state_values[non_final_mask] = target(non_final_next_states, mean_only=True).max(1)[0]
                policy_vec = Variable(torch.Tensor([0.5, 0, 0, 0.5]))
                next_state_values[non_final_mask] = (policy_vec*target(non_final_next_states, mean_only=True)).sum(1)
                

                # Now, we don't want to mess up the loss with a volatile flag, so let's
                # clear it. After this, we'll just end up with a Variable that has
                # requires_grad=False
                next_state_values.volatile = False
                # Compute the expected Q values
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch
                
                ## See https://arxiv.org/pdf/1608.05081.pdf for loss function
                loss_components[3] += (state_action_values - expected_state_action_values).pow(2).sum()

                #Now add log(q(w|theta)) - log(p(w)) term
                mu_l = model.get_mu_l()
                sigma_l = model.get_sigma_l()
                sigma_p_l = model.get_sigma_p_l()
                
                for i in range(len(w_sample)):
                    w = w_sample[i]
                    mu = mu_l[i]
                    sigma = sigma_l[i]
                    sigma_p = sigma_p_l[i]
                    loss_components[0] -= torch.log(sigma).sum()
                    loss_components[1] += (w.pow(2)).sum() / (2.0 * (sigma_p ** 2))
                    loss_components[2] -= ((w - mu).pow(2) / (2.0 * sigma.pow(2))).sum()
                
                for i in range(len(loss_components)):
                    average_loss_components[i] += loss_components[i]
                loss = sum(loss_components)
              
                average_loss += loss
                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
            
                # for param in model.parameters():
                #     param.grad.data.clamp_(-1, 1)
                optimizer.step()
    for i in range(len(loss_components)):
        average_loss_components[i] /= (NUM_TARGET_RESET * NUM_EPOCHS * NUM_SAMPLES)
        loss_comp_dict[loss_component_names[i]].append(average_loss_components[i].data[0])

    average_loss /= (NUM_TARGET_RESET * NUM_EPOCHS * NUM_SAMPLES)
    loss_list.append(average_loss.data[0])
    memory.clear()
 


######################################################################
#
# Below, you can find the main training loop. At the beginning we reset
# the environment and initialize the ``state`` variable. Then, we sample
# an action, execute it, observe the next screen and the reward (always
# 1), and optimize our model once.

def policy_dump():
    size = env.size
    for i in range(size):
        for j in range(size):
            state = Tensor((i,j)).unsqueeze(0)
            action = select_action(state)
            print "State: {}; Action: {}".format(state[0], action[0,0])

def Q_dump():
    size = env.size
    for i in range(size):
        for j in range(size):
            state = Tensor((i,j)).unsqueeze(0)
            Q = get_Q(state)
            print "State: {}; Q: {}".format(state[0], Q)

total_score_l = []
sample_period = 5
num_episodes = 50000
starting_uncertainty = model.get_sigma_l()

sigma_average_dict = defaultdict(list)
components = ['W: first hidden', 'b: first hidden', 'W: second hidden', 'b: second hidden', \
                'W: output','b: output']

for i_episode in range(num_episodes):
    # Initialize the environment and state
    env.reset()
    state = Tensor(env.get_state()).unsqueeze(0)
    score = 0

    for t in xrange(500):
        # Select and perform an action
        if t % sample_period == 0:
            w_sample = model.sample()
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
        if done:
            total_score_l.append(score)
            break
    # Perform one step of the optimization (on the target network)
    if i_episode > 0 and i_episode % 20 == 0 :
        optimize_model()
        print "Iter {},\tscore: {},\tloss:{}".\
                    format(i_episode, score, loss_list[-1])
        for idx, sigma in enumerate(model.get_sigma_l()):
            average = sigma.mean().data[0]
            sigma_average_dict[components[idx]].append(average)


ending_uncertainty = model.get_sigma_l()

# policy_dump()
Q_dump()

# plt.hist(correlations, bins=100)
# plt.xlabel("Normalized dot product")
# plt.ylabel("Number of pairs of sampled gradients")
# plt.title("Sampled gradient similarity")
# plt.savefig("dot_product_gradients_removed.png")
# plt.show()

# N = 100
# averages = np.convolve(total_score_l, np.ones((N,))/N, mode='valid')
# plt.plot(xrange(len(averages)), averages)
# plt.show()

print "Average score: {}".format(sum(total_score_l)/float(len(total_score_l)))
print "Starting uncertainty:"
print starting_uncertainty
print "Ending uncertainty:"
print ending_uncertainty
print "Ending means:"
print model.get_mu_l()

handles = []
for name, values in sigma_average_dict.iteritems():
    handle, = plt.plot(values, label=name)
    handles.append(handle)
plt.legend(handles=handles)
plt.xlabel("Number of updates")
plt.ylabel("Mean of uncertainty in each layer")
plt.title("Uncertainty versus number of updates")
plt.savefig("ADAM_decay_of_average_sigma.png")
plt.show()

handles = []
for name, values in loss_comp_dict.iteritems():
    handle, = plt.plot(values, label=name)
    handles.append(handle)
plt.legend(handles=handles)
plt.xlabel("Number of updates")
plt.ylabel("Average loss contribution")
plt.title("Loss contribution by term versus number of updates")
plt.savefig("ADAM_loss_contributions.png")
plt.show()


plt.plot(loss_list)
plt.xlabel("Number of updates")
plt.ylabel("Average loss")
plt.title("Average loss versus number of updates")
plt.savefig("ADAM_loss.png")
plt.show()
