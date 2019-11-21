import copy
import random
from collections import namedtuple

import numpy as np, gym
import torch
import torch.nn as nn
import torch.nn.functional as F

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Actor, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        output = torch.sigmoid(self.layer3(output))
        # output = torch.tanh(self.layer3(output))
        return output


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, num_actions):
        super(Critic, self).__init__()

        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, num_actions)

    def forward(self, state):
        output = F.relu(self.layer1(state))
        output = F.relu(self.layer2(output))
        output = self.layer3(output)
        return output


Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class ReplayBuffer:

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
        # if batch_size > len(self):
        #     return self.memory
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# Ornstein-Uhlenbeck Process
# Taken from https://github.com/vitchyr/rlkit/blob/master/rlkit/exploration_strategies/ou_strategy.py
class OUNoise(object):
    def __init__(self, action_space, mu=0.0, theta=0.15, max_sigma=0.3,
                 min_sigma=0.3, decay_period=100000):
        self.mu = mu
        self.theta = theta
        self.sigma = max_sigma
        self.max_sigma = max_sigma
        self.min_sigma = min_sigma
        self.decay_period = decay_period
        self.action_dim = action_space.shape[0]
        self.low = action_space.low
        self.high = action_space.high
        self.reset()

    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu

    def evolve_state(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(
            self.action_dim)
        self.state = x + dx
        return self.state

    def get_action(self, action, t=0):
        ou_state = self.evolve_state()
        self.sigma = self.max_sigma - (self.max_sigma - self.min_sigma) * min(
            1.0, t / self.decay_period)
        return np.clip(action + ou_state, self.low, self.high)


class DDPGAgent:

    def __init__(self, env, actor_net, critic_net,
                 replay_buffer, tau, actor_lr = 1e-4, critic_lr = 1e-3,
                 gamma=0.95, training_reset_steps = 50):
        self.env = env
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_net = copy.deepcopy(actor_net)
        self.actor_target_net = copy.deepcopy(actor_net)
        self.critic_net = copy.deepcopy(critic_net)
        self.critic_target_net = copy.deepcopy(critic_net)
        self.replay_buffer = replay_buffer
        self.training_reset_steps = training_reset_steps
        self.tau = tau

    def fit_actor_output_to_env(self, action):
        # Below assumes action between 0 and 1, which is right for sigmoid
        return (self.env.action_space.high - self.env.action_space.low) * action

    def train_nn(self, discount, batch_size, print_Q=False):

        lossFunction = nn.MSELoss()
        critic_optimizer = torch.optim.Adam(self.critic_net.parameters(), lr=self.critic_lr)
        actor_optimizer = torch.optim.Adam(self.actor_net.parameters(),
                                            lr=self.actor_lr)

        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        batch = np.array(batch)

        states = batch[:,0]
        states = np.stack(states)
        actions = torch.Tensor(np.array(batch[:,1], dtype=np.float))
        rewards = torch.Tensor(np.array(batch[:,2], dtype=np.float))
        next_states = batch[:,3]
        next_states = np.stack(next_states)
        dones = torch.Tensor((np.array(batch[:,4]==0, dtype=np.float)))

        # We want the gradient on this pass so we can update with backprop afterwards
        predicted_Q = self.Q(states, actions, self.critic_net, no_grad=False)

        target_next_actions = self.act(next_states, self.actor_target_net, no_grad=True)

        target_next_Q = self.Q(next_states, target_next_actions, self.critic_target_net, no_grad=True)

        targets = (rewards + discount * target_next_Q.squeeze() * dones)

        targets = targets.view(-1, 1)

        critic_loss = lossFunction(predicted_Q, targets)

        critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_optimizer.step()

        actions = self.act(states, self.actor_net, no_grad=False)
        Q_vals = self.Q(states, actions, self.critic_net, no_grad=False)

        if print_Q:
            print((Q_vals.mean()).item())
        # Negative is important! Higher value = lower loss
        actor_loss = -Q_vals.mean()

        actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_optimizer.step()

    def soft_update(self, target_net, curr_net):
        for target_param, curr_param in zip(target_net.parameters(), curr_net.parameters()):
            target_param.data.copy_(self.tau * curr_param.data + (1-self.tau) * target_param.data)

    def train(self, num_steps, batch_size):
        state = self.env.reset()

        noise = OUNoise(self.env.action_space)

        for step in range(num_steps):

            action = self.act(state, self.actor_net)
            action = noise.get_action(action.detach().numpy(), step)
            action = self.fit_actor_output_to_env(action)
            next_state, reward, done, info = self.env.step(action)

            self.replay_buffer.push(state, action, reward, next_state, done)

            print_Q = False
            # if step % 150 == 149:
            #     print_Q = True
            self.train_nn(self.gamma, batch_size=batch_size, print_Q=print_Q)

            self.soft_update(self.critic_target_net, self.critic_net)

            self.soft_update(self.actor_target_net, self.actor_net)

            state = next_state

            # Return early if done
            if done:
                return

    def Q(self, states, actions, neural_net_to_use, no_grad = False):
        states = torch.from_numpy(states)
        states = states.float()

        actions = actions.view(-1, 1)

        s_a_pairs = torch.cat((states, actions), dim=1)

        if no_grad:
            with torch.no_grad():
                output = neural_net_to_use(s_a_pairs)
            return output

        output = neural_net_to_use(s_a_pairs)
        return output

    def maxQ(self, states, neural_net_to_use):
        # The no_grad=True line below is very important on this part!
        # Why? Well when we backprop, we only want the forward pass to count
        # towards the autogradient/diff once
        maxQ = self.Q(states, neural_net_to_use, no_grad=True)
        maxQ = torch.max(maxQ, dim=1)

        return maxQ

    def forward_pass(self, neural_net, state, no_grad=True):
        if no_grad:
            with torch.no_grad():
                output = neural_net(state)
        else:
            output = neural_net(state)
        return output

    def act(self, state, neural_net_to_use, no_grad=True):
        state = torch.FloatTensor(state)
        actions = self.forward_pass(neural_net_to_use, state, no_grad)
        return actions

    def run_episode(self, max_steps):
        """ Runs a test episode """
        state = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            next_state, reward, done, _ = self.env.step(
                self.fit_actor_output_to_env(self.act(state, self.actor_net).detach().numpy()))

            state = next_state
            total_reward += reward

        return total_reward


def visualize(episodes: int):

    for _ in range(episodes):
        state = env.reset()

        for _ in range(max_env_steps):
            env.render()
            new_state, reward, done, info = env.step(
                agent.fit_actor_output_to_env(agent.act(state, agent.actor_net).detach().numpy()))
            state = new_state


max_env_steps = 300
test_episodes = 100
train_episodes = 5
replay_buffer_capacity = 5000000000
tau = 0.001 * 5
# Smaller learning rate helps avoid unlearning.

replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

env = gym.make('Pendulum-v0')
initial = env.reset()

# print(initial)
# print(env.action_space.low)
# print(env.action_space.high)

num_actions = 1

critic_net = Critic(input_size=len(initial) + num_actions, hidden_size=128, num_actions=num_actions)
actor_net = Actor(input_size=len(initial), hidden_size=128, num_actions=num_actions)


agent = DDPGAgent(env, actor_net, critic_net, replay_buffer, tau=tau)

batch_size = 128 * 5


print("Max env steps: " + str(max_env_steps))
print("Test_episodes: " + str(test_episodes))
print("Train episodes: " + str(train_episodes))
print("Replay buffer capacity: " + str(replay_buffer_capacity))
print("Batch size: " + str(batch_size))
print("Tau : " + str(tau))


for loop_iter in range(100):

    print()
    print("Loop iter: " + str(loop_iter + 1))

    for training_episode in range(train_episodes):
        # print("Training episode: " + str(training_episode + 1))
        agent.train(max_env_steps, batch_size=batch_size)

    total_reward = 0
    for _ in range(test_episodes):
        total_reward += agent.run_episode(max_steps=max_env_steps)
    print("Average reward: " + str(np.round(total_reward / test_episodes, 1)))

    visualize(1)


