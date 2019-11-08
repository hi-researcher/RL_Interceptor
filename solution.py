from game import Interceptor_V2
from game.Interceptor_V2 import Init, Draw, Game_step, World
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import random

gamma = 0.99

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.distributions import Categorical


class InterceptorNetwork(nn.Module):
    def __init__(self):
        super(InterceptorNetwork, self).__init__()

        self.number_of_actions = 4

        self.relu = nn.ReLU(inplace=True)
        self.pool = torch.nn.MaxPool2d(kernel_size=4)
        self.softmax = nn.Softmax(dim=1)

        self.conv1 = nn.Conv2d(3, 4, 2)
        self.conv2 = nn.Conv2d(4, 4, 2)
        self.conv3 = nn.Conv2d(4, 8, 2)
        self.conv4 = nn.Conv2d(8, 8, 2)
        self.conv5 = nn.Conv2d(8, 16, 2)

        self.fc4 = nn.Linear(24, 15)
        self.fc5 = nn.Linear(16, self.number_of_actions)

    def forward(self, x, ang):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv2(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv3(out)
        out = self.relu(out)
        out = self.pool(out)

        out = self.conv4(out)
        out = self.relu(out)
        out = self.pool(out)

        # out = self.conv5(out)
        # out = self.relu(out)
        # out = self.pool(out)

        out = out.view(out.size()[0], -1)
        out = self.fc4(out)
        out = self.relu(out)

        out = torch.cat((out, ang), 1)

        out = self.fc5(out)

        out = self.softmax(out)

        return out


class Agent():
    def __init__(self, resolution=10):
        self.model = InterceptorNetwork()
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-6)
        self.criterion = nn.MSELoss()

        self.reward_history = []
        self.loss_history = []
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []

        # self.iterations = iterations
        self.epsilon = 0.1
        self.final_epsilon = 0.0001
        # self.epsilon_decrese = (self.epsilon - self.final_epsilon)/self.iterations
        self.gamma = 0.99
        self.replay_memory_size = 10000

        self.resolution = resolution

    def create_model_state(self, game_state):
        res = self.resolution

        image = np.zeros((3, int(World.width/res), int(World.height/res)))

        r_locs, i_locs, c_locs, ang = game_state

        phase = np.array([[5000, 0]])
        if len(r_locs) > 0:
            image[0, np.int((r_locs + phase)/res)] = 1
        if len(i_locs) > 0:
            image[1, np.int((i_locs + phase)/res), 1] = 1
        for c_loc in c_locs:
            image[2, 0, int((5000 + c_loc[0])/res): int((5000 + c_loc[0] + c_loc[1])/res)] = 1

        return torch.Tensor(image[np.newaxis, :, :, :]), torch.tensor([[ang]], dtype=torch.float)

    def get_action(self, game_state=None):
        # if random.random() <= self.epsilon:
        #     print("Performed random action!")
        #     return np.random.randint(0, 4)
        self.state = game_state

        # transform state to model representation
        model_state, ang = self.create_model_state(game_state)

        # Select an action
        output = self.model(model_state, ang)
        c = Categorical(output)
        action = c.sample()

        # Add log probability of our chosen action to our history
        if self.policy_history.dim() != 0:
            self.policy_history = torch.cat([self.policy_history, c.log_prob(action)])
        else:
            self.policy_history = (c.log_prob(action))

        return action

    def update_reward(self, reward):
        self.reward_episode.append(reward)

    def update_policy(self):
        R = 0
        rewards = []
        # Discount future rewards back to the present using gamma
        for r in self.reward_episode[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = (torch.sum(torch.mul(self.policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.loss_history.append(float(loss))
        self.reward_history.append(np.sum(self.reward_episode))
        self.policy_history = torch.autograd.Variable(torch.Tensor())
        self.reward_episode = []


if __name__ == '__main__':
    total_episodes = 5000  # Set total number of episodes to train agent on.
    iterations_per_episode = 1000
    short_memory = 50
    myAgent = Agent()

    for episode in range(total_episodes):
        Init()
        state = ([], [], [], 0)
        step = 0
        running_reward = None
        last_score = 0
        for short_episode in range(int(iterations_per_episode / short_memory)):

            for time in range(short_memory):
                action_button = myAgent.get_action(state)

                r_locs, i_locs, c_locs, ang, score = Game_step(action_button)

                myAgent.update_reward(score - last_score)
                last_score = score
                # Draw()

            myAgent.update_policy()

            print(f'Episode {episode}\tShort Episode {short_episode}\tRunning Reward: {myAgent.reward_history[-1]}\tScore: {score}')
