import copy
import random

import numpy as np, gym
import torch
import torch.nn as nn
import torch.nn.functional as F


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size)
        self.layer2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        output = F.relu(self.layer1(x))
        output = self.layer2(output)
        return output


class ReplayBuffer:

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, transition_tuple):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(transition_tuple)
        else:
            self.memory[self.position] = transition_tuple
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class QLearningAgent:

    def __init__(self, env, neural_net, replay_buffer, lr=0.01, gamma=0.99, training_reset_steps = 50):
        self.env = env
        self.gamma = gamma
        self.lr = lr
        self.q_being_updated = copy.deepcopy(neural_net)
        self.q_target_net = copy.deepcopy(neural_net)
        self.replay_buffer = replay_buffer
        self.training_reset_steps = training_reset_steps

    def train_nn(self, discount, batch_size):

        lossFunction = nn.MSELoss()
        optimizer = torch.optim.Adam(self.q_being_updated.parameters(), lr=self.lr)

        if len(self.replay_buffer) < batch_size:
            return

        batch = self.replay_buffer.sample(batch_size)
        batch = np.array(batch)

        states = np.stack(batch[:,0])
        actions = torch.Tensor(np.array(batch[:,1], dtype=np.float))
        rewards = torch.Tensor(np.array(batch[:,2], dtype=np.float))
        next_states = np.stack(batch[:,3])
        dones = torch.Tensor((np.array(batch[:,4]==0, dtype=np.float)))

        predicted_Q = self.Q(states, self.q_being_updated)

        reshaped_predicted_Q = predicted_Q.gather(1, actions.long().view(-1,1))

        next_states_maxQ = self.maxQ(next_states, self.q_target_net).values

        targets = (rewards + discount * next_states_maxQ * dones)

        targets = targets.view(-1, 1)
        loss = lossFunction(reshaped_predicted_Q, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, batch_size, num_steps=100000):
        state = self.env.reset()

        eps_start = 0.9
        eps_end = 0.1
        eps_decay = 100

        for step in range(num_steps):
            epsilon = eps_end + (eps_start - eps_end) * (np.math.exp(- step / eps_decay))

            action = self.act(state, epsilon=epsilon)

            next_state, reward, done, info = self.env.step(action)

            self.replay_buffer.push((state, action, reward, next_state, done))

            self.train_nn(self.gamma, batch_size)
            if step % self.training_reset_steps == self.training_reset_steps - 1:
                # self.q_target_net = copy.deepcopy(self.q_being_updated)
                self.q_target_net.load_state_dict(self.q_being_updated.state_dict())

            state = next_state

            if done:
                return
            # Can we return early if done? Yeah.

    def Q(self, states, neural_net_to_use, no_grad = False):
        """This function returns, for a set of states as input, a set of
        sets of Q values as output, where each subset contains the Q values
        across all possible actions. Then we use max or argmax to extract
        either the desired Q value or the desired action to take
        """

        states = torch.from_numpy(states)
        states = states.float()

        if no_grad:
            with torch.no_grad():
                output = neural_net_to_use(states)
            return output

        output = neural_net_to_use(states)
        return output

    def maxQ(self, states, neural_net_to_use):
        # The no_grad=True line below is very important on this part!
        # Why? Well when we backprop, we only want the forward pass to count
        # towards the autogradient/diff once
        maxQ = self.Q(states, neural_net_to_use, no_grad=True)
        maxQ = torch.max(maxQ, dim=1)

        return maxQ

    def act(self, state, epsilon=0.):

        # Epsilon-greedy action
        if np.random.random() < epsilon:
            return self.env.action_space.sample()

        best_action = torch.argmax(self.Q(state, self.q_being_updated, no_grad=True))

        return best_action.item()

    def run_episode(self, max_steps=200):
        """ Runs a test episode """
        state = self.env.reset()
        total_reward = 0
        for step in range(max_steps):
            next_state, reward, done, _ = self.env.step(self.act(state))

            state = next_state
            total_reward += reward

            if done:
                break

        return total_reward


def visualize(episodes: int):

    for _ in range(episodes):
        state = env.reset()

        for _ in range(max_env_steps):

            env.render()
            new_state, reward, done, info = env.step(agent.act(state))
            state = new_state

            if done:
                break


# 200 steps standard I guess
max_env_steps = 200
test_episodes = 10
train_episodes = 25
learning_rate = 1e-3
# Smaller learning rate helps avoid the unlearning.

batch_size = 200 * 5

replay_buffer_capacity = 10000*1000


print("Max env steps: " + str(max_env_steps))
print("Test_episodes: " + str(test_episodes))
print("Train episodes: " + str(train_episodes))
print("Replay buffer capacity: " + str(replay_buffer_capacity))
print("Batch size: " + str(batch_size))


neural_net = NeuralNet(input_size=4, hidden_size=10, output_size=2)

replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

env = gym.make('CartPole-v1')
initial = env.reset()

agent = QLearningAgent(env, neural_net, replay_buffer, lr=learning_rate)

for loop_iter in range(100):

    print()
    print("Loop iter: " + str(loop_iter + 1))

    for training_episode in range(train_episodes):
        # print("Training episode: " + str(training_episode))
        agent.train(max_env_steps, batch_size)

    total_reward = 0
    for _ in range(test_episodes):
        total_reward += agent.run_episode(max_steps=max_env_steps)
    print("Average reward: " + str(np.round(total_reward / test_episodes, 1)))

    visualize(1)


