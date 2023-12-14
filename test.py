import gymnasium as gym
import torch
import torch.optim as optim
from torchsummary import summary
import numpy as np
import os
import matplotlib.pyplot as plt
from itertools import count
import math
import time
import datetime
import logging
import json

from envi import DescreteEnv, ContinuousEnv
from models import DQN, ResNet18DQN
from buffer import ReplayBuffer
from train import train
from utils import save_returngraph, plot_durations, save_model, check_dir


def main():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    test_mode = False
    is_continuous = False
    is_preprocess = True
    stack_frames = 4
    learning_rate = 0.0005
    gamma = 0.98  
    batch_size = 128
    epsilon_init = 1.0
    epsilon_min = 0.1
    epsilon_cycle = 2000
    n_episode = 2000
    buffer_save = 2000
    buffer_limit = 50000        # size of replay buffer
    print_interval = 20
    max_episode_steps = 1000

    # model save
    # return: 6.509인 모델
    dir_path = './models'
    modelname = f'CarRacing_DQN_8.907.pt'
    model_path = os.path.join(dir_path, modelname)


    # random seed 설정
    random_seed = 42
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed) # if use multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)


    print('test start')

    # env = gym.make("CarRacing-v2", render_mode = 'state_pixels', continuous = is_continuous)
    env = gym.make("CarRacing-v2", render_mode = 'human', continuous = is_continuous)
    
    # env setting   
    env = DescreteEnv(env, stack_frames = stack_frames,  is_preprocess = is_preprocess)
    
    # env reset
    s, _  = env.reset()        # s.shape = (4, 84, 84)
    state_dim = s.shape
    action_dim = env.action_space.n 

    ## 2. model
    q = DQN(state_dim[0], action_dim, random_seed).to(device)
    
    optimizer = optim.RMSprop(q.parameters(), learning_rate)

    checkpoint = torch.load(model_path)
    q.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])

    # 2) env 초기화
    # s, _  = env.reset() 
    score = 0 
    time_start = time.time()

    for t in range(0, max_episode_steps) :
        t += 1
        epsilon = 0
        # transition 만들기: (s, a, r, s', done_mask)
        action_space_mapping = None
        a = q.sample_action(torch.from_numpy(s).float().to(device), epsilon, is_continuous, action_space_mapping)
        s_prime, r, terminated, truncated, info = env.step(a)

        # r: 작은 수로 만들기
        r = r/100.0      
        
        # done : 0 -> terminal
        done = (terminated or truncated) 
        done_mask = 0.0 if done else 1.0   

        transition = (s, a, r, s_prime, done_mask)
        
        # memory.put(tuple(transition))

        # 다음 state로 가기
        s = s_prime

        score  += r
        # print(t, a, score)

        if done:
            break
    time_end = time.time()
    sec = time_end - time_start
    times = str(datetime.timedelta(seconds=sec)).split(".")[0]
       
if __name__ == "__main__":
    main()