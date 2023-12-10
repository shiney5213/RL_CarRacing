import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import torch
import torch.optim as optim


from envi import SetEnv


# https://github.com/wpiszlogin/driver_critic/blob/main/base_solution.py참고
# 모델 변경(DDpG -> DQN), framework 변경(tensorflow -> torch)
# https://gymnasium.farama.org/environments/box2d/car_racing/
# https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html


class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, activation = F.relu):
        super(DQN, self).__init__()

        # model
        # input : (N, 4, 84, 84)
        # output: (N, C_out, H_out, W_out)
        self.conv1 = nn.Conv2d(in_channels = state_dim,       # [N, 4, 84, 84 ]
                                out_channels = 16,    # [N, 16, 20, 20]
                                kernel_size = 8,
                                stride = 8) 
        self.conv2 = nn.Conv2d(16, 32, 4, 2)          # [N, 16, 20, 20] -> [N, 32, 9, 9]
        self.in_features = 32 * 4 * 4        # 512
        self.fc1 = nn.Linear(self.in_features, 256)
        self.fc2 = nn.Linear(256, action_dim)
        self.activation = activation

    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon):
        out = self.forward(obs)    # torch.Size([1, 5])
        random_value = random.random()

        # if random_value < epsilon :
        #     return [random.randint(1,5) for i in range(0,5)]

        # else:
        #     return [ out[0].argmax().item(), out[1].argmax().item(), out[2].argmax().item(), out[3].argmax().item(),out[4].argmax().item(),]

        if random_value < epsilon :
            return random.randint(0,4) 

        else:
            return  out.argmax().item()