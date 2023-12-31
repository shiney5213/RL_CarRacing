import gymnasium as gym
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import random
import collections
import torch
import torch.optim as optim
from torchvision.models import resnet18, ResNet18_Weights



# https://github.com/wpiszlogin/driver_critic/blob/main/base_solution.py참고
# 모델 변경(DDpG -> DQN), framework 변경(tensorflow -> torch)
# 공식 문서 : https://gymnasium.farama.org/environments/box2d/car_racing/
# https://hiddenbeginner.github.io/study-notes/contents/tutorials/2023-04-20_CartRacing-v2_DQN.html
# https://github.com/Jueun-Park/gym-autonmscar/tree/master
# 평가 지표: https://davinci-ai.tistory.com/33
# 슈퍼마리오 예제: https://brunch.co.kr/@kakao-it/144
# tensorflow CarRacing: https://github.com/andywu0913/OpenAI-GYM-CarRacing-DQN/tree/master
# resnet18 model : https://towardsdatascience.com/applying-a-deep-q-network-for-openais-car-racing-game-a642daf58fc9

# models.py
class ResNet18DQN(nn.Module):
    # https://towardsdatascience.com/applying-a-deep-q-network-for-openais-car-racing-game-a642daf58fc9
    def __init__(self, state_dim, action_dim, random_seed):
        super(ResNet18DQN, self).__init__()
        self.action_dim = action_dim
        
        np.random.seed(random_seed)
        
        
        self.resnet18_pretrained =resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.in_features = self.resnet18_pretrained.fc.in_features
        self.resnet18_pretrained.fc = nn.Linear(self.in_features, 512)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, self.action_dim)

    def forward(self, x):
        x = self.resnet18_pretrained(x)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon, is_continuous, action_space_mapping):
        obs = obs.unsqueeze(0)  # [1, 4, 84, 84]
        out = self.forward(obs)    # torch.Size([1, 5]) / torch.Size([1, 3])
        random_value = random.random()

        if random_value < epsilon :
            # ramdon action
            if is_continuous:
                random_num = random.randint(0, self.action_dim-1)
                return action_space_mapping[random_num]
            else:
                return random.randint(0, self.action_dim-1) 

        else:
            if is_continuous:
                action_num = out.argmax().item()
                return action_space_mapping[action_num]
                  
            else:
                return  out.argmax().item()

class DQN(nn.Module):
    def __init__(self, state_dim, action_dim, random_seed):
        super(DQN, self).__init__()
        self.action_dim = action_dim
        
        np.random.seed(random_seed)
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

    
    def forward(self, x):

        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view((-1, self.in_features))
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon, is_continuous, action_space_mapping= None):
        out = self.forward(obs)    # torch.Size([1, 5]) / torch.Size([1, 3])
        random_value = random.random()

        if random_value < epsilon :
            # ramdon action
            if is_continuous:
                random_num = random.randint(0, self.action_dim-1)
                return action_space_mapping[random_num]
            else:
                return random.randint(0, self.action_dim-1) 

        else:
            if is_continuous:
                action_num = out.argmax().item()
                return action_space_mapping[action_num]
                  
            else:
                return  out.argmax().item()


class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, random_seed):
        super(DeepQNetwork, self).__init__()
        self.action_dim = action_dim
        np.random.seed(random_seed)

        self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size= 8, stride = 4), nn.ReLU(inplace = True))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size= 4, stride = 2), nn.ReLU(inplace = True))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= 3, stride = 1), nn.ReLU(inplace = True))

        self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64, 512), nn.ReLU(inplace = True))
        self.fc2 = nn.Linear(512, 5)
        # self.conv1 = nn.Sequential(nn.Conv2d(4, 32, kernel_size= 8, stride = 4), nn.ReLU(inplace = True))
        # self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size= 4, stride = 2), nn.ReLU(inplace = True))
        # self.conv3 = nn.Sequential(nn.Conv2d(64, 64, kernel_size= 3, stride = 1), nn.ReLU(inplace = True))

        # self.fc1 = nn.Sequential(nn.Linear(7 * 7 * 64 , 512), nn.ReLU(inplace = True))
        # self.fc2 = nn.Linear(512, 5)
        self._create_weights()


    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        in_feature =  64 * 7 *  7   # 6272
        x = x.view(-1, in_feature)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

    def sample_action(self, obs, epsilon, is_continuous, action_space_mapping= None):
        out = self.forward(obs)    # torch.Size([1, 5]) / torch.Size([1, 3])
        random_value = random.random()

        if random_value < epsilon :
            # ramdon action
            if is_continuous:
                random_num = random.randint(0, self.action_dim-1)
                return action_space_mapping[random_num]
            else:
                return random.randint(0, self.action_dim-1) 

        else:
            if is_continuous:
                action_num = out.argmax().item()
                return action_space_mapping[action_num]
                  
            else:
                return  out.argmax().item()
